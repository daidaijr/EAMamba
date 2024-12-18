import os
import cv2
import random

import numpy as np

import torch
import math

def calc_padding(img_h, img_w, base=64):
    pad_h = 0 if img_h % base == 0 else base - (img_h % base)
    pad_w = 0 if img_w % base == 0 else base - (img_w % base)

    pad_t = 0
    pad_b = pad_h

    pad_l = 0
    pad_r = pad_w

    if pad_h == 0 and pad_w == 0:
        return None
        
    return (pad_l, pad_t, pad_r, pad_b)

def read_img(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def resize_img(img, out_h, out_w):
    return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

# from restormer (some images are smaller than the largest crop size)
def padding(img_gt, img_lq, gt_size):
    h, w, _ = img_gt.shape

    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)
    
    if h_pad == 0 and w_pad == 0:
        return img_gt, img_lq

    #! USE BORDER REFLECT
    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    # print('img_lq', img_lq.shape, img_gt.shape)
    if img_lq.ndim == 2:
        img_lq = np.expand_dims(img_lq, axis=2)
    if img_gt.ndim == 2:
        img_gt = np.expand_dims(img_gt, axis=2)
    return img_gt, img_lq

def data_augmentation(img, mode):
    if mode == 0:
        out = img
    elif mode == 1:
        out = np.flipud(img)
    elif mode == 2:
        out = np.rot90(img)
    elif mode == 3:
        out = np.rot90(img)
        out = np.flipud(out)
    elif mode == 4:
        out = np.rot90(img, k=2)
    elif mode == 5:
        out = np.rot90(img, k=2)
        out = np.flipud(out)
    elif mode == 6:
        out = np.rot90(img, k=3)
    elif mode == 7:
        out = np.rot90(img, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid augmentation')
    
    return out

def random_augmentation(img):
    aug_mode = random.randint(0, 7)

    out = data_augmentation(img, aug_mode).copy()

    return out

def paired_random_augmentation(img_1, img_2):
    aug_mode = random.randint(0, 7)

    out_1 = data_augmentation(img_1, aug_mode).copy()
    out_2 = data_augmentation(img_2, aug_mode).copy()

    return out_1, out_2
