import cv2
import random

import numpy as np

import torch
from torch.utils.data import Dataset

from torchvision import transforms

import datasets
import scipy.io as sio

from datasets import register
from datasets.data_utils import calc_padding , read_img, rgb2ycbcr, paired_random_augmentation, padding
from datasets.data_utils import PairedRandomCrop

@register('paired-dataset')
class PairedDataset(Dataset):
    def __init__(
        self,
        gt_data_file,
        lq_data_file,
        phase='train',
        crop_size=128,
        repeat=1,
        augment=False,
        y_only=False,
    ):
        super(PairedDataset, self).__init__()

        self.gt_paths = [p.strip() for p in open(gt_data_file)]
        self.lq_paths = [p.strip() for p in open(lq_data_file)]

        assert len(self.gt_paths) == len(self.lq_paths)

        self.phase = phase
        self.repeat = repeat
        self.augment = augment
        self.y_only = y_only
        self.crop_size = crop_size

        if phase != 'test':
            self.pre_process = PairedRandomCrop(self.crop_size)

    def __len__(self):
        return len(self.gt_paths) * self.repeat

    def __getitem__(self, idx):
        return_dict = {}

        # Read Image
        img_gt_path = self.gt_paths[idx % len(self.gt_paths)]
        img_lq_path = self.lq_paths[idx % len(self.lq_paths)]

        img_gt = read_img(img_gt_path)
        img_lq = read_img(img_lq_path)
        if self.y_only:
            img_gt = np.expand_dims(rgb2ycbcr(img_gt, y_only=True), -1)
            img_lq = np.expand_dims(rgb2ycbcr(img_lq, y_only=True), -1)

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

        return_dict['gt'] = patch_gt
        return_dict['lq'] = patch_lq

        return return_dict

@register('dualpixel-paired-dataset')
class DualPixelPairedDataset(Dataset):
    def __init__(
        self,
        gt_data_file,
        lql_data_file,
        lqr_data_file,
        phase='train',
        crop_size=128,
        repeat=1,
        augment=False,
        y_only=False,
    ):
        super(DualPixelPairedDataset, self).__init__()

        self.gt_paths = [p.strip() for p in open(gt_data_file)]
        self.lql_paths = [p.strip() for p in open(lql_data_file)]
        self.lqr_paths = [p.strip() for p in open(lqr_data_file)]

        assert len(self.gt_paths) == len(self.lql_paths) and len(self.gt_paths) == len(self.lqr_paths)

        # file client (io backend)
        self.phase = phase
        self.repeat = repeat
        self.augment = augment
        self.y_only = y_only
        self.crop_size = crop_size

        if phase != 'test':
            self.pre_process = PairedRandomCrop(self.crop_size)

    def __len__(self):
        return len(self.gt_paths) * self.repeat

    def __getitem__(self, idx):
        return_dict = {}

        # Read Image
        img_gt_path = self.gt_paths[idx % len(self.gt_paths)]
        img_lql_path = self.lql_paths[idx % len(self.lql_paths)]
        img_lqr_path = self.lqr_paths[idx % len(self.lqr_paths)]

        img_gt = read_img(img_gt_path)
        img_lql = read_img(img_lql_path)
        img_lqr = read_img(img_lqr_path)
    
        if self.y_only:
            img_gt = np.expand_dims(rgb2ycbcr(img_gt, y_only=True), -1)
            img_lql = np.expand_dims(rgb2ycbcr(img_lql, y_only=True), -1)
            img_lqr = np.expand_dims(rgb2ycbcr(img_lqr, y_only=True), -1)

        # concate both lq at channel dimension after y_only expansion
        img_lq = np.concatenate((img_lql, img_lqr), 2)   # h x w x 6

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
        patch_lq = transforms.ToTensor()(patch_lq) # 6 x h x w

        h, w = patch_gt.shape[-2:]
        to_pad = calc_padding(h, w, base=8)

        if to_pad is not None:
            patch_lq = transforms.Pad(padding=to_pad, padding_mode='reflect')(patch_lq) # 0, 0, right, bottom
            return_dict['padding'] = torch.IntTensor( [round(p) for p in to_pad] )

        return_dict['gt'] = patch_gt
        return_dict['lq'] = patch_lq

        return return_dict