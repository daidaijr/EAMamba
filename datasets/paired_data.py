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
    
@register('sr-paired-dataset')
class SRPairedDataset(Dataset):
    def __init__(
        self,
        gt_data_file,
        lq_data_file,
        phase='train',
        crop_size=128,
        repeat=1,
        augment=False,
        resize_back=False,
        y_only=False,
        **kwargs
    ):
        super(SRPairedDataset, self).__init__()

        self.gt_paths = [p.strip() for p in open(gt_data_file)]
        self.lq_paths = [p.strip() for p in open(lq_data_file)]

        assert len(self.gt_paths) == len(self.lq_paths)

        self.phase = phase
        self.repeat = repeat
        self.augment = augment
        self.crop_size = crop_size
        self.resize_back = resize_back
        self.y_only = y_only

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

        scale_factor = img_gt.shape[0] // img_lq.shape[0]

        if self.phase == 'test':
            h_lr, w_lr = img_lq.shape[:2]

            patch_lq = img_lq
            patch_gt = img_gt[:h_lr*scale_factor, :w_lr*scale_factor, :]
        else:
            hr_crop_size = self.crop_size
            lr_crop_size = hr_crop_size // scale_factor

            h_lr, w_lr = lr_crop_size, lr_crop_size

            # Ensure image dimensions are at least as large as the crop size
            if img_lq.shape[0] < lr_crop_size or img_lq.shape[1] < lr_crop_size:
               pad_h = max(0, lr_crop_size - img_lq.shape[0])
               pad_w = max(0, lr_crop_size - img_lq.shape[1])
               img_lq = np.pad(img_lq, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
               pad_h_hr = pad_h * scale_factor
               pad_w_hr = pad_w * scale_factor
               img_gt = np.pad(img_gt, ((0, pad_h_hr), (0, pad_w_hr), (0, 0)), mode='reflect')

            if self.phase == 'train':
                x0 = random.randint(0, img_lq.shape[0] - h_lr)
                y0 = random.randint(0, img_lq.shape[1] - w_lr)
            else:
                x0 = (img_lq.shape[0] - h_lr) // 2
                y0 = (img_lq.shape[1] - w_lr) // 2

            patch_lq = img_lq[x0: x0+h_lr, y0: y0+w_lr, :]

            x1 = x0 * scale_factor
            y1 = y0 * scale_factor

            patch_gt = img_gt[x1: x1+hr_crop_size, y1: y1+hr_crop_size, :]

        h_hr, w_hr = patch_gt.shape[:2]

        # Augmentation
        if self.augment:
            patch_gt, patch_lq = paired_random_augmentation(patch_gt, patch_lq)

        patch_gt = transforms.ToTensor()(patch_gt) # 3 x h x w
        patch_lq = transforms.ToTensor()(patch_lq) # 3 x h x w

        if self.resize_back:
            patch_lq = resize_fn(patch_lq, h_hr, w_hr)
            h_lr, w_lr = h_hr, w_hr
            scale_factor = 1

        to_pad = calc_padding(h_lr, w_lr, base=8)

        if to_pad is not None:
            patch_lq = transforms.Pad(padding=to_pad, padding_mode='reflect')(patch_lq) # left, top, right, bottom
            return_dict['padding'] = torch.IntTensor( [p * scale_factor for p in to_pad] )

        return_dict['gt'] = patch_gt
        return_dict['lq'] = patch_lq

        return return_dict
