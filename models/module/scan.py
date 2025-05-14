import math
import numpy as np

import torch

ALLOWED_SCAN_TYPE = [None, 'diagonal', 'zorder', 'zigzag', 'hilbert']
ALLOWED_SCAN_MERGE_METHOD = ['add', 'concate']
ALLOWED_SCAN_COUNT = [1, 2, 4, 8]

class ScanTransform():
    '''
        Transforming tensor with different scan method, using pre-computed indices
    '''
    def __init__(self, scan_type, scan_count, merge_method) -> None:
        assert scan_type in ALLOWED_SCAN_TYPE, f'{scan_type} is not in allowed scan type : {ALLOWED_SCAN_TYPE}'
        assert scan_count in ALLOWED_SCAN_COUNT, f'{scan_count} is not in allowed scan count : {ALLOWED_SCAN_COUNT}'
        if scan_count > 1:     # only need to merge when more than one scan direction
            assert merge_method in ALLOWED_SCAN_MERGE_METHOD, \
            f'{merge_method} is not in allowed scan merge method : {ALLOWED_SCAN_MERGE_METHOD}'

        self.scan_type = scan_type
        self.scan_count = scan_count
        self.merge_method = merge_method

        if self.scan_type == 'diagonal':
            self.scan_method = self.diagonal_scan
        elif self.scan_type == 'zorder':
            self.scan_method = self.z_order_scan
        elif self.scan_type == 'zigzag':
            self.scan_method = self.zigzag_scan
        elif self.scan_type == 'hilbert':
            self.scan_method = self.hilbert_scan
        else:
            self.scan_method = None

        self.index_dict, self.invert_index_dict = {}, {}

    def diagonal_scan(self, size):
        height, width = size

        indices = np.arange(height * width).reshape(height, width)
        result = []
        
        for sum_idx in range(height + width - 1):
            start_row = max(0, sum_idx - width + 1)
            end_row = min(sum_idx + 1, height)
            diagonal = [indices[i, sum_idx - i] for i in range(start_row, end_row)]
            result.extend(diagonal)
        
        return np.array(result)

    def hilbert_scan(self, size):
        from hilbert import encode
        
        height, width = size

        # Determine the next power of 2 greater than or equal to the max dimension
        max_dim = 2 ** int(np.ceil(np.log2(max(height, width))))
        
        # Generate Hilbert curve indices for a max_dim x max_dim grid
        coords = np.array([[i, j] for i in range(max_dim) for j in range(max_dim)])
        p = int(np.log2(max_dim))
        indices = encode(coords, 2, p)
        
        # Sort the indices and filter out the ones within the desired grid size
        sorted_indices = np.argsort(indices)
        sorted_coords = coords[sorted_indices]
        
        # Filter valid coordinates and convert to 1D indices
        valid_mask = (sorted_coords[:, 0] < height) & (sorted_coords[:, 1] < width)
        valid_sorted_coords = sorted_coords[valid_mask]
        valid_sorted_indices = valid_sorted_coords[:, 0] * width + valid_sorted_coords[:, 1]
        
        return valid_sorted_indices.tolist()

    def zigzag_scan(self, size):
        height, width = size
        
        # max_dim = max(height, width)
        a = np.arange(height * width).reshape(height, width)
        # zigzag_indices = np.concatenate([np.diagonal(a[::-1,:], k)[::(2*(k % 2)-1)] for k in range(1 - max_dim, max_dim)])
        zigzag_indices = np.concatenate([np.diagonal(a[::-1,:], k)[::(2*(k % 2)-1)] for k in range(1 - height, width)])
        return zigzag_indices

    def z_order_scan(self, size):
        height, width = size
            
        # Create a grid of coordinates
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        y, x = y.flatten(), x.flatten()

        # Vectorized Morton code calculation
        def interleave_bits(x, y):
            MAGIC = torch.tensor([0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF], dtype=torch.int32)
            def part1by1(n):
                n = (n | (n << 8)) & MAGIC[4]
                n = (n | (n << 4)) & MAGIC[3]
                n = (n | (n << 2)) & MAGIC[2]
                n = (n | (n << 1)) & MAGIC[1]
                return n
            x = part1by1(x)
            y = part1by1(y)
            return x | (y << 1)

        morton_codes = interleave_bits(x, y)

        # Sort the flattened tensor according to the Morton codes
        sorted_indices = torch.argsort(morton_codes)
        return sorted_indices

    # Flipping and stacking
    def tensor_reorder(self, x, size):
        '''
            Reorder the tensor by flipping and stacking

            input : B, C, L 
            output:
                add mode      : B, count, C, L          -> copy at channel, reorder and stack at batch dim
                concate mode  : B, count, C/count, L    -> split at channel, reorder and stack at batch dim
        '''
        if self.scan_count == 1:
            return x

        # copy the tensor in the channel dim for self.scan_count times
        # and stack them together in batch dim later
        if self.merge_method == 'add':
            # B, C, L -> B, count*C, L
            x = x.repeat(1, self.scan_count, 1)

        H, W = size
        B, C, L = x.shape
        
        to_stack = []
        sections = torch.split(x, C // self.scan_count, dim=1)
        for i, section in enumerate(sections):
            if i in range(2, len(sections), 4) or i in range(3, len(sections), 4):      
                if self.scan_type is None or i >= 4: # To 4D and swap H and W 
                    section = section.view(B, -1, H, W).permute(0, 1, 3, 2).reshape(B, -1, L)
                else: # To 4D and flip W 
                    section = torch.flip(section.view(B, -1, H, W), (3,)).reshape(B, -1, L)
            if i % 2 == 1:  
                # Flip the 2nd and 4th section
                section = torch.flip(section, (2,))
            to_stack.append(section)

        return torch.stack(to_stack, dim=1)
    
    # Reverse function of tensor_reorder
    def tensor_restore(self, x, size):
        '''
            Restore the order of the tensor by reverse flipping
            
            input : 
                add mode      : B, count, C, L
                concate mode  : B, count, C/count, L
            output:
                B, C, L 
        '''
        if self.scan_count == 1:
            return x

        H, W = size
        B, _, C, L = x.shape

        to_stack = []
        sections = torch.split(x, 1, dim=1)
        for i, section in enumerate(sections):
            if i % 2 == 1:  # Reverse flip the section
                section = torch.flip(section, (3,))
            if i in range(2, len(sections), 4) or i in range(3, len(sections), 4):
                if self.scan_type is None or i >= 4: # To 4D and swap H and W 
                    section = section.view(B, 1, C, W, H).permute(0, 1, 2, 4, 3).reshape(B, 1, C, L)
                else: # To 4D and flip W 
                    section = torch.flip(section.view(B, 1, C, H, W), (4,)).reshape(B, 1, C, L)

            to_stack.append(section)

        x = torch.cat(to_stack, dim=1)

        if self.merge_method == 'add':
            x = x.view(B, self.scan_count, C, L).sum(dim=1)
        elif self.merge_method == 'concate':
            x = x.view(B, self.scan_count, C, L).reshape(B, self.scan_count * C, L)
        return x
    
    # Reorder and apply the pre-computed scan index
    def apply_scan_transform(self, x, size):
        if x is None:    # default route or z branch disabled
            return x

        # Apply tensor reorder
        x = self.tensor_reorder(x, size)   

        if self.scan_count == 8:
            x[:, :4, ...] = x[:, :4, :, self.get_entry(size)]
        else:
            x = x[:, :, :, self.get_entry(size)] if self.scan_type is not None else x

        return x

    # Reverse function of apply_scan_transform
    def restore_scan_transform(self, x, size):
        # Invert the scan index
        if self.scan_type is not None:
            if self.scan_count == 8:
                x[:, :4, ...] = x[:, :4, :, self.get_entry(size, get_invert=True)]
            else:
                x = x[:, :, :, self.get_entry(size, get_invert=True)] 

        # Invert the tensor reorder
        return self.tensor_restore(x, size)

    # Create and store the scan index if not existed yet, else return the scan index
    def get_entry(self, size, get_invert=False):
        key = str(size)

        # if the resulotion hasn't been seen before
        if self.index_dict.get(key) is None:
            # Apply the scan method
            index = self.scan_method(size)
            # Register the entry
            # self.index_dict[key] = index
            if isinstance(index, torch.Tensor):
                self.index_dict[key] = index.clone().detach()
            else:
                self.index_dict[key] = torch.tensor(index, dtype=torch.long)

            # Create invert index for re-mapping to original order
            invert_index = [0] * len(index)
            for i, idx in enumerate(index):
                invert_index[idx] = i
            # self.invert_index_dict[key] = invert_index
            if isinstance(invert_index, torch.Tensor):
                self.invert_index_dict[key] = invert_index.clone().detach()
            else:
                self.invert_index_dict[key] = torch.tensor(invert_index, dtype=torch.long)
                
            return index if not get_invert else invert_index
        else:
            return self.index_dict[key] if not get_invert else self.invert_index_dict[key]
