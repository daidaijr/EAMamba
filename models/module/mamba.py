import math

from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

# Extended Mamba block with custom modifications
class ExtendedMamba(nn.Module):
    def __init__(
        self,
        d_model,
        scan_transform,
        use_checkpoint=False,           # enable checkpoint to trade speed with GPU memory
        conv_2d=False,
        disable_z_branch=False,         # disable the gating branch
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        layer_idx=None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        
        self.scan_transform = scan_transform
        self.scan_type = scan_transform.scan_type
        self.scan_count = scan_transform.scan_count
        self.scan_merge_method = scan_transform.merge_method

        self.conv_2d = conv_2d

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        self.disable_z_branch = disable_z_branch
        in_proj_output_size = self.d_inner * 2 if not disable_z_branch else self.d_inner 
        self.in_proj = nn.Linear(self.d_model, in_proj_output_size, bias=bias, **factory_kwargs)

        # conv 1d or 2d
        self.conv = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=self.d_inner,
            **factory_kwargs,
        ) if conv_2d else \
        nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            **factory_kwargs,
        )

        self.act = nn.SiLU()

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # Reduce part of the dim size if merge method is concate (count*B, (1/count)C, L)
        if self.scan_merge_method == 'concate':
            self.d_inner = self.d_inner // self.scan_count

        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs) 
            for _ in range(self.scan_count)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs) for _ in range(self.scan_count)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        # S4D real initialization
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.scan_count, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=self.scan_count, merge=True)

        # Restore part of the dim size 
        # if self.scan_merge_method == 'concate':
        #     self.d_inner = self.d_inner * self.scan_count       

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward(self, hidden_states, x_size):
        """
            hidden_states: (B, L, D) 
            x_size: (H, W)

            Returns: same shape as hidden_states
        """

        batch, seqlen, dim = hidden_states.shape
        h, w = x_size

        def in_proj_function(hidden_states):
            # We do matmul and transpose BLD -> DBL at the same time
            xz = rearrange(
                self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
                "d (b l) -> b d l",
                l=seqlen,
            )
            if self.in_proj.bias is not None:
                xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
            
            return xz

        # 1. in_proj
        if self.use_checkpoint:
            xz = checkpoint(in_proj_function, hidden_states, use_reentrant=False)
        else:
            xz = in_proj_function(hidden_states)

        if not self.disable_z_branch:
            x, z = xz.chunk(2, dim=1)       # z is the branched part
        else:
            x = xz
            z = None

        # 2. conv
        if self.conv_2d:
            x = rearrange(x, 'b d (h w) -> b d h w', h=h, w=w)
        
        if self.use_checkpoint:
            x = checkpoint(self.conv, x, use_reentrant=False)
        else:
            x = self.conv(x)

        if self.conv_2d:
            x = rearrange(x, 'b d h w -> b d (h w)')
        else:
            x = x[..., :seqlen]
        
        x = self.act(x)

        # 3. selective scan (SSM)
        # Custom scan direction 
        x = self.scan_transform.apply_scan_transform(x, x_size)

        k = self.scan_count
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(batch, k, -1, seqlen), self.dt_projs_weight)
        xs = x.float().view(batch, -1, seqlen)
        dts = dts.contiguous().float().view(batch, -1, seqlen)
        Bs = Bs.float().view(batch, k, -1, seqlen)
        Cs = Cs.float().view(batch, k, -1, seqlen)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        y = selective_scan_fn(
                xs, dts,
                As, Bs, Cs, Ds, z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                return_last_state=False,
            ).view(batch, k, -1, seqlen)

        y = self.scan_transform.restore_scan_transform(y, x_size)

        # 4. out_proj
        y = rearrange(y, "b d l -> b l d")
        y = self.out_norm(y)
        if not self.disable_z_branch:
            z = rearrange(z, "b d l -> b l d")
            y = y * F.silu(z)
        if self.use_checkpoint:
            out = checkpoint(self.out_proj, y, use_reentrant=False)
        else:
            out = self.out_proj(y)
        
        return out