## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

# Modification :
#   Customized for our own models.
#   Removed restormer model load section
#   Removed the import from basicsr local package

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage import img_as_ubyte
import h5py
import scipy.io as sio

# local package
import models
# import utils
# from .. import models

parser = argparse.ArgumentParser(description='Generate SIDD .mat file for matlab evaluation')
parser.add_argument('--input_dir', default='datasets/SIDD/test/SIDD', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./SIDD_results/', type=str, help='Directory for results')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--model', required=True)

args = parser.parse_args()

##########################

result_dir_mat = os.path.join(args.result_dir, 'mat')
os.makedirs(result_dir_mat, exist_ok=True)

if args.save_images:
    result_dir_png = os.path.join(args.result_dir, 'png')
    os.makedirs(result_dir_png, exist_ok=True)

# Load the model
model_spec = torch.load(args.model)['model']
model_restoration, _ = models.make(model_spec, load_sd=True)
model_restoration.cuda().eval()

model_dir = os.path.dirname(args.model)
model_name = os.path.basename(args.model).replace(".pth", "").replace(".pt", "")
model_name = model_name.replace('current_iter', os.path.basename(model_dir))

# Process data
filepath = os.path.join(args.input_dir, 'ValidationNoisyBlocksSrgb.mat')
img = sio.loadmat(filepath)
Inoisy = np.float32(np.array(img['ValidationNoisyBlocksSrgb']))
Inoisy /=255.
restored = np.zeros_like(Inoisy)
with torch.no_grad():
    for i in tqdm(range(40)):
        for k in range(32):
            # 40, 32, 256, 256, 3 (.mat) -> 1, 3, 256, 256 (tensor)
            noisy_patch = torch.from_numpy(Inoisy[i,k,:,:,:]).unsqueeze(0).permute(0, 3, 1, 2).cuda()
            restored_patch = model_restoration(noisy_patch)
            # 1, 3, 256, 256 (tensor) -> 256, 256, 3 (to be saved in .mat)
            restored_patch = torch.clamp(restored_patch,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
            restored[i,k,:,:,:] = restored_patch

            if args.save_images:
                save_file = os.path.join(result_dir_png, '%04d_%02d.png'%(i+1,k+1))
                utils.save_img(save_file, img_as_ubyte(restored_patch))

# save denoised data
sio.savemat(os.path.join(result_dir_mat, f'{model_name}_Idenoised.mat'), {"Idenoised": restored,})
