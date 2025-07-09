import argparse
import math
import yaml

from fvcore.nn import FlopCountAnalysis, flop_count_table

import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import models
import utils

def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)

def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex

    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops

def selective_scan_flop_jit(inputs, outputs, flops_fn=flops_selective_scan_fn, verbose=False):
    if verbose:
        print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)

    return flops

def register_supported_ops():
    supported_ops = {
        "aten::gelu": None,
        "aten::silu": None,
        "aten::leaky_relu_": None,
        "aten::neg": None,
        "aten::exp": None,
        "aten::flip": None,
        "aten::pixel_shuffle": None,
        "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,
    }

    return supported_ops

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--var', action='store_true', default=False)    # False: 256*256, True: 32*32 -> 1024*1024 step = 2x
    parser.add_argument('--simple', action='store_true', default=False) # To output whole model info or not
    args = parser.parse_args()

    various = args.var
    simple_output = args.simple

    if various:
        print('\nRunning in various resolution mode, will only stop after CUDA OOM.')
    else:
        print('\nRunning in standard mode (256x256).')

    if simple_output:
        print('\nSimplified output, full model won\'t be printed.')
    else:
        print('\nFull model info will be printed.')

    # if various resolution, start at 32*32
    current_resolution = 32 if various else 256

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model, model_e = models.make(config['model'])
    model.cuda()

    if not simple_output:
        print(f'\nmodel: #struct={model}')
    print(f'model: #params={utils.compute_num_params(model, text=True)}')

    supported_ops = register_supported_ops()

    while True:
        inputs = torch.randn(1, 3, current_resolution, current_resolution).cuda()

        flops = FlopCountAnalysis(model, (inputs, ))
        flops_info = flop_count_table(
            flops.set_op_handle(**supported_ops), 
            max_depth=5,
            show_param_shapes=False 
        )

        print(f'\nCurrent resolution : {current_resolution} x {current_resolution}')
        print(f'Total : {flops.total()/(10**9) :.2f} GFLOPs')
        print(f'Max memory allocated : {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB')
        
        if not simple_output:
            print(f'\n{flops_info}')

        if not various:
            break
        else:
            current_resolution *= 2

    print('\nDone')
        