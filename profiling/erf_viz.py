import random
import argparse
import os
from tqdm import tqdm

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from timm.utils import AverageMeter

import torch
import torch.nn as nn
from torchvision import transforms

# packages in project directory
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import models
from datasets.data_utils.transforms import CenterCropForERF

# dataset_path = "datasets/list/Denoise/Urban100.txt"
# dataset_path = "datasets/list/SR/RealSRx4_valid_gt.txt"
# dataset_path = "datasets/list/Derain/Test1200_lq.txt"
dataset_path = 'datasets/list/Denoise/SIDD_valid_lq.txt'

# dataset_name = 'urban'
# dataset_name = 'RealSRx4'
# dataset_name = 'Test1200_lq'
dataset_name = 'sidd_lq'

# the size for the tested patch
crop_image_size = 128

# Stop the process after N images (For debug) (max : 100 for urban100 & realSR, 1280 for SIDD)
early_stop_count = 100

# position
h_option = 'c'  # 'c', 't', 'b'
w_option = 'c'  # 'c', 'l', 'r'

# noise amount (max=255) (need to pass --noise flag)
noise_amount = 50

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # sets the seed for cpu
    torch.cuda.manual_seed(seed)  # Sets the seed for the current GPU.
    torch.cuda.manual_seed_all(seed)  #  Sets the seed for the all GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plt_setting():
    plt.rcParams['font.family'] = "Times New Roman"
    plt.rcParams['font.family'] = "Times New Roman"

    params = {
        'axes.titlesize': 24,
        'legend.fontsize': 24,
        # 'figure.figsize': (16, 10),
        'axes.labelsize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'figure.titlesize': 24,
        'savefig.dpi': 300,
    }

    plt.rcParams.update(params)
    plt.style.use('seaborn-whitegrid')

    sns.set_style('white')
    plt.rcParams['axes.unicode_minus'] = False

def calcGrads(model, inputs):
    outputs = model(inputs)

    # 1, 3, 128, 128
    outputs_shape = outputs.size()
    if w_option == 'c':
        position_w = outputs_shape[3] // 2  # center
    elif w_option == 'l':
        position_w = 10                     # left
    elif w_option == 'r':
        position_w = outputs_shape[3] - 10  # right
    else:
        assert False, "Position w not defined"

    if h_option == 'c':
        position_h = outputs_shape[2] // 2  # center
    elif h_option == 't':
        position_h = 10                     # top
    elif h_option == 'b':
        position_h = outputs_shape[2] - 10  # bottom
    else:
        assert False, "Position h not defined"

    # (center is just a single number)
    center = outputs[..., position_h, position_w].sum()

    # tuple of only one tensor
    grads = torch.autograd.grad(center, inputs)
    # 1, 3, 128, 128
    grads = grads[0]

    # grads = F.relu(grads)
    grads = torch.abs(grads)
    # print(grads)
    
    # 128, 128
    aggregated_grads = grads.sum( (0, 1) )
    # print(aggregated_grads)

    return aggregated_grads.cpu().numpy()

def visualize_erf(erf_grads, save_path='erf_heatmap.png'):
    print(np.max(erf_grads))
    print(np.min(erf_grads))

    ax = None
    norm_fn = lambda x: np.power(x - 1, 0.25)
    viz_grads = norm_fn(erf_grads + 1)
    viz_grads = viz_grads / np.max(viz_grads)

    plt.figure()
    ax = sns.heatmap(
        viz_grads,
        xticklabels=False,
        yticklabels=False,
        cmap='RdYlGn',
        center=0, annot=False, ax=ax, cbar=True, annot_kws={'size': 24}, fmt='.2f',
        square=True
    )
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path)

def read_img(img_path):
    return transforms.ToTensor()(
        transforms.ToPILImage()(
            transforms.ToTensor()(
                Image.open(img_path).convert('RGB')
            )
        )
    )

def read_image_list(file_path):
    with open(file_path, 'r') as file:
        img_list = [line.strip() for line in file if line.strip()]
    return img_list

if __name__ == '__main__':

    img_list = read_image_list(dataset_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--noise', action='store_true', default=False)    # For DN 
    args = parser.parse_args()

    set_seed(2454)

    plt_setting()

    add_noise = args.noise

    # Load the model
    model_spec = torch.load(args.model)['model']

    model, model_e = models.make(model_spec, load_sd=True)

    model.cuda().eval()

    # Get the directory of the model
    model_dir = os.path.dirname(args.model)
    model_name = os.path.basename(args.model).replace(".pth", "").replace(".pt", "")
    model_name = model_name.replace('current_iter', os.path.basename(model_dir))

    optimizer = torch.optim.SGD(model.parameters(), lr=0, weight_decay=0)
    grads_meter = AverageMeter()

    length = len(img_list)
    center_crop_fn = CenterCropForERF(crop_image_size)

    count = 0
    for i in tqdm(range(length)):
        img_path = img_list[i]

        ##### Prepare Data #####
        hq_img = read_img(img_path)
        hq_img = center_crop_fn(hq_img)
        
        if add_noise:
            noise = torch.from_numpy( np.float32(np.random.randn( *(hq_img.shape) )) * noise_amount / 255. )
            hq_img = torch.clamp(hq_img + noise, 0, 1)

        hq_img = (hq_img * 2.0 - 1.0).cuda()

        inp = hq_img.unsqueeze(0)

        ##### ERF Calculation #####
        inp.requires_grad = True
        optimizer.zero_grad()
        contribution_scores = calcGrads(model, inp)
        torch.cuda.empty_cache()
        if np.isnan( np.sum(contribution_scores) ):
            print('Got NaN, continue')
            continue
        else:
            grads_meter.update(contribution_scores)

        count += 1
        if count >= early_stop_count:
            break

    save_path = os.path.join(model_dir, f"{model_name}_{crop_image_size}_{dataset_name}_{early_stop_count}_erf_{h_option}_{w_option}.png")
    erf_grads = grads_meter.avg
    visualize_erf(erf_grads, save_path)
