import cv2
import math
import random

import numpy as np

import torch
from torch.utils.data import Dataset

from torchvision import transforms

import datasets

from datasets import register
from datasets.data_utils import PairedRandomCrop
from datasets.data_utils import read_img, paired_random_augmentation, padding, calc_padding

@register('denoising-color-dataset')
class DenoisingColorDataset(Dataset):
    def __init__(
        self,
        gt_data_file,
        phase='train',
        crop_size=128,
        repeat=1,
        augment=False,
        sigma_type="constant",
        sigma_range=15,         # num 15,25,50 or list [0, 50] (for random)
    ):
        super(DenoisingColorDataset, self).__init__()

        self.gt_paths = [p.strip() for p in open(gt_data_file)]

        self.phase = phase
        self.repeat = repeat
        self.augment = augment
        self.crop_size = crop_size
        self.sigma_type = sigma_type
        self.sigma_range = sigma_range
        assert self.sigma_type in ['constant', 'random', 'choice']

        if phase != 'test':
            self.pre_process = PairedRandomCrop(self.crop_size)

    def __len__(self):
        return len(self.gt_paths) * self.repeat

    def __getitem__(self, idx):
        return_dict = {}

        # Read Image
        img_gt_path = self.gt_paths[idx % len(self.gt_paths)]

        img_gt = read_img(img_gt_path)
        img_lq = img_gt.copy()

        if self.phase == 'train':
            img_gt, img_lq = padding(img_gt, img_lq, self.crop_size)

        if self.phase != 'test':
            patch_gt, patch_lq = self.pre_process(img_gt, img_lq)
        else:
            patch_gt, patch_lq = img_gt, img_lq

        # Augmentation
        if self.augment:
            patch_gt, patch_lq = paired_random_augmentation(patch_gt, patch_lq)

        patch_gt = transforms.ToTensor()(patch_gt) # 3 x h x w
        patch_lq = transforms.ToTensor()(patch_lq) # 3 x h x w
        
        h, w = patch_gt.shape[-2:]
        to_pad = calc_padding(h, w, base=8)

        if to_pad is not None:
            patch_lq = transforms.Pad(padding=to_pad, padding_mode='reflect')(patch_lq) # 0, 0, right, bottom
            return_dict['padding'] = torch.IntTensor( [round(p) for p in to_pad] )


        #! Noisify for denoising
        if self.sigma_type == 'constant':
            sigma_value = self.sigma_range
        elif self.sigma_type == 'random':
            sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
        elif self.sigma_type == 'choice':
            sigma_value = random.choice(self.sigma_range)
        else:   # default to constant
            sigma_value = self.sigma_range
        
        if self.phase == 'train':
            #! Original algorithm used in Restormer
            noise_level = torch.FloatTensor([sigma_value])/255.0
            noise = torch.randn(patch_lq.size()).mul_(noise_level).float()
            patch_lq.add_(noise)
        else:
            #! Restormer do this section in np, and convert to tensor later
            pre_seed = torch.seed()
            torch.manual_seed(2454)
            # In restormer, there's an additional "sigma_test" config setting
            # which means the type of noise value is always constant
            # I've modified it so that it can also be random
            noise_level = torch.FloatTensor([sigma_value])/255.0
            noise = torch.randn(patch_lq.size()).mul_(noise_level).float()
            patch_lq.add_(noise)
            torch.manual_seed(pre_seed)
        
        return_dict['gt'] = patch_gt
        return_dict['lq'] = patch_lq

        return return_dict