import argparse
import pathlib
import time
import yaml

from functools import partial
from tqdm import tqdm
import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from einops import rearrange

import datasets
import models
import utils
from PIL import Image


def rgb2ycbcr(img, y_only=False):
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.

    return out_img


def calc_psnr(pred, gt, y_only=False, rgb_range=1.):
    if y_only:
        pred = rgb2ycbcr(pred, y_only=True)
        gt = rgb2ycbcr(gt, y_only=True)

    diff = pred - gt
    mse = diff.pow(2).mean()

    return 20 * torch.log10(rgb_range / mse.sqrt())


def calc_ssim(pred, gt, y_only=False, rgb_range=1.):
    if y_only:
        pred = rgb2ycbcr(pred, y_only=True)
        gt = rgb2ycbcr(gt, y_only=True)

    pred = pred * 255. / rgb_range
    gt = gt * 255. / rgb_range

    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11).expand(pred.size(1), 1, 11, 11).to(pred.dtype).to(pred.device)
    
    mu1 = F.conv2d(pred, window, groups=pred.shape[1])[..., 5:-5, 5:-5]  # valid mode
    mu2 = F.conv2d(gt, window, groups=gt.shape[1])[..., 5:-5, 5:-5]  # valid mode
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, groups=pred.shape[1])[..., 5:-5, 5:-5] - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, window, groups=gt.shape[1])[..., 5:-5, 5:-5] - mu2_sq
    sigma12 = F.conv2d(pred * gt, window, groups=pred.shape[1])[..., 5:-5, 5:-5] - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

    return ssim_map.mean()


def forward_patch(model, img, shave=16., crop_size=256, scale=1):
    min_size = crop_size*crop_size
    b, c, h, w = img.size()

    top = slice(0, int(h // 2 + shave - (h // 2 % 8)))
    bottom = slice(int(h - h // 2 - shave + ((h - h // 2) % 8)), h)
    left = slice(0, int(w // 2 + shave - (w // 2 % 8)))
    right = slice(int(w - w // 2 - shave + ((w - w // 2) % 8)), w)
    x_chops = [img[..., top, left],
        img[..., top, right],
        img[..., bottom, left],
        img[..., bottom, right]
    ]

    y_chops = []
    if h * w < min_size:
        for i in range(0, 4):
            y_chops.append(model.forward(x_chops[i]))
    else:
        y_chops = [
            forward_patch(model, patch, shave=shave, crop_size=crop_size) for patch in x_chops
        ]

    h *= scale
    w *= scale
    top = slice(0, h//2)
    bottom = slice(h - h//2, h)
    bottom_r = slice(h//2 - h, None)
    left = slice(0, w//2)
    right = slice(w - w//2, w)
    right_r = slice(w//2 - w, None)

    y = img.new(b, c, h, w)
    y[..., top, left] = y_chops[0][..., top, left]
    y[..., top, right] = y_chops[1][..., top, right_r]
    y[..., bottom, left] = y_chops[2][..., bottom_r, left]
    y[..., bottom, right] = y_chops[3][..., bottom_r, right_r]

    return y


def evaluate(
    loader,
    model,
    name=None,
    eval_y_only=False,
    eval_crop_size=None,
    ensemble=False,
    save_dir=None,
    current_iter=None,
    verbose=False,
    save_image=False,
    scale=1
):
    model.eval()

    if save_dir:
        save_dir = pathlib.Path(save_dir)
        tmp_dir = name + '-results'
        img_dir = save_dir / tmp_dir
        img_dir.mkdir(parents=True, exist_ok=True)

    val_psnr = utils.Averager()
    val_ssim = utils.Averager()
    metric_psnr = partial(calc_psnr, y_only=eval_y_only)
    metric_ssim = partial(calc_ssim, y_only=eval_y_only)

    val_time = utils.Averager()
    pbar = tqdm(loader, leave=False, desc='val')

    IDX = 1

    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()
        lq = batch['lq']
        bs = batch['lq'].shape[0]
        lqs = [lq]
        if ensemble:
            lqs.extend([lq.flip(-1), lq.flip(-2), lq.flip(-1, -2)])

        torch.cuda.synchronize()
        start = time.time()

        preds = []
        with torch.no_grad():
            for lq in lqs:
                if eval_crop_size is None:
                    preds.append(model(lq))
                else:
                    preds.append(forward_patch(model, lq, crop_size=eval_crop_size, scale=scale))
        end = time.time()
        torch.cuda.synchronize()
        val_time.add(end - start, bs)

        # crop to GT's size
        gt = batch['gt']
        hq_h, hq_w = gt.shape[-2:]
        lq = lq[..., :hq_h, :hq_w]
        if ensemble:
            pred = (preds[0] + preds[1].flip(-1) + preds[2].flip(-2) + preds[3].flip(-1, -2)) / 4
        else:
            pred = preds[0]
        pred = pred[..., :hq_h, :hq_w]
        pred = torch.clip(pred, 0, 1)

        if scale > 1:
            res_psnr = metric_psnr(pred[:, :, scale:-scale, scale:-scale], gt[:, :, scale:-scale, scale:-scale])
            res_ssim = metric_ssim(pred[:, :, scale:-scale, scale:-scale], gt[:, :, scale:-scale, scale:-scale])
        else:
            res_psnr = metric_psnr(pred, gt)
            res_ssim = metric_ssim(pred, gt)
        val_psnr.add(res_psnr, bs)
        val_ssim.add(res_ssim, bs)

        if save_dir:
            if name == 'validation':  # only save last batch
                if lq.shape[-1] == gt.shape[-1]:    # Only if the data has the same shape                    
                    # B C H W -> C B*H W
                    final_pred = rearrange(pred, 'b c h w -> c (b h) w')
                    final_gt = rearrange(gt, 'b c h w -> c (b h) w')
                    lq_tensor = rearrange(lq, 'b c h w -> c (b h) w')
                    saved_tensor = torch.cat((lq_tensor, final_pred, final_gt), dim=2)
                    saved_image = transforms.ToPILImage()(saved_tensor.cpu())
                    saved_image.save(img_dir / f'current_iter-{str(current_iter)}.png')
            else:
                if save_image:
                    saved_image = transforms.ToPILImage()(pred[0].cpu())
                    saved_image.save(img_dir / f'{str(IDX).zfill(4)}.png')
                IDX += 1
                with open(img_dir / 'PSNR.txt', mode='a') as f:
                    print('result: {:.6f} | time: {:.6f}'.format(res_psnr.item(), end - start), file=f)
                with open(img_dir / 'SSIM.txt', mode='a') as f:
                    print('result: {:.6f} | time: {:.6f}'.format(res_ssim.item(), end - start), file=f)

        if verbose:
            des = 'val avg psnr: {:.4f} ssim {:.4f}'.format(val_psnr.item(), val_ssim.item())
            pbar.set_description(des)

    if save_dir and name != 'validation':
        with open(img_dir / 'PSNR.txt', mode='a') as f:
            print('AVG-result: {:.6f}'.format(val_psnr.item()), file=f)
            print('AVG-Time: {:.6f}'.format(val_time.item()), file=f)
        with open(img_dir / 'SSIM.txt', mode='a') as f:
            print('AVG-result: {:.6f}'.format(val_ssim.item()), file=f)
            print('AVG-Time: {:.6f}'.format(val_time.item()), file=f)

    return val_psnr.item(), val_ssim.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--dataset')
    parser.add_argument('--ensemble', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)   # save images or not
    args = parser.parse_args()
    
    sv_file = torch.load(args.model)
    model_spec = sv_file['model']
    print(sv_file['current_iter'])

    dataset_dict = {
        'RealSRx2': ['configs/test/sr/test-RealSRx2.yaml'],
        'RealSRx3': ['configs/test/sr/test-RealSRx3.yaml'],
        'RealSRx4': ['configs/test/sr/test-RealSRx4.yaml'],
        'SIDD': ['configs/test/denoise/test-SIDD.yaml'],
        'Denoise': ['configs/test/denoise/test-CBSD68-color-sig15.yaml',
                    'configs/test/denoise/test-Kodak24-color-sig15.yaml',
                    'configs/test/denoise/test-McMaster-color-sig15.yaml',
                    'configs/test/denoise/test-Urban100-color-sig15.yaml',
                    'configs/test/denoise/test-CBSD68-color-sig25.yaml', 
                    'configs/test/denoise/test-Kodak24-color-sig25.yaml',
                    'configs/test/denoise/test-McMaster-color-sig25.yaml',
                    'configs/test/denoise/test-Urban100-color-sig25.yaml',
                    'configs/test/denoise/test-CBSD68-color-sig50.yaml',
                    'configs/test/denoise/test-Kodak24-color-sig50.yaml',
                    'configs/test/denoise/test-McMaster-color-sig50.yaml',
                    'configs/test/denoise/test-Urban100-color-sig50.yaml'],
        'Denoise-sig15': ['configs/test/denoise/test-CBSD68-color-sig15.yaml',
                          'configs/test/denoise/test-Kodak24-color-sig15.yaml',
                          'configs/test/denoise/test-McMaster-color-sig15.yaml',
                          'configs/test/denoise/test-Urban100-color-sig15.yaml'],
        'Denoise-sig25': ['configs/test/denoise/test-CBSD68-color-sig25.yaml',
                          'configs/test/denoise/test-Kodak24-color-sig25.yaml',
                          'configs/test/denoise/test-McMaster-color-sig25.yaml',
                          'configs/test/denoise/test-Urban100-color-sig25.yaml'],
        'Denoise-sig50': ['configs/test/denoise/test-CBSD68-color-sig50.yaml',
                          'configs/test/denoise/test-Kodak24-color-sig50.yaml',
                          'configs/test/denoise/test-McMaster-color-sig50.yaml',
                          'configs/test/denoise/test-Urban100-color-sig50.yaml'],
        'Motion_Deblur': ['configs/test/deblur/test-GoPro.yaml',
                          'configs/test/deblur/test-HIDE.yaml'],
        'GoPro': ['configs/test/deblur/test-GoPro.yaml'],
        'HIDE' : ['configs/test/deblur/test-HIDE.yaml'],
        'RESIDE_ITS': ['configs/test/dehaze/test-RESIDE-ITS.yaml'],
        'RESIDE_OTS': ['configs/test/dehaze/test-RESIDE-OTS.yaml'], 
        'Derain': ['configs/test/derain/test-Rain100H.yaml',
                    'configs/test/derain/test-Rain100L.yaml',
                    'configs/test/derain/test-Test100.yaml',
                    'configs/test/derain/test-Test1200.yaml',
                    'configs/test/derain/test-Test2800.yaml'],
        'SRx2': ['configs/test/sr/test-ySet5x2.yaml',
                 'configs/test/sr/test-ySet14x2.yaml',
                 'configs/test/sr/test-yB100x2.yaml',
                 'configs/test/sr/test-yUrban100x2.yaml'],
        'SRx3': ['configs/test/sr/test-ySet5x3.yaml',
                 'configs/test/sr/test-ySet14x3.yaml',
                 'configs/test/sr/test-yB100x3.yaml',
                 'configs/test/sr/test-yUrban100x3.yaml'],
        'SRx4': ['configs/test/sr/test-ySet5x4.yaml',
                 'configs/test/sr/test-ySet14x4.yaml',
                 'configs/test/sr/test-yB100x4.yaml',
                 'configs/test/sr/test-yUrban100x4.yaml'],
        'DPDD': ['configs/test/deblur/test-DPDD-indoor.yaml',
                 'configs/test/deblur/test-DPDD-outdoor.yaml',
                 'configs/test/deblur/test-DPDD.yaml'],
    }

    test_list = dataset_dict[args.dataset]
    
    _, model_e = models.make(model_spec, load_sd=True)
    model_e = model_e.cuda()

    for config_name in test_list:
        with open(config_name, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        spec = config['test_dataset']
            
        dataset = datasets.make(spec['dataset'])
        loader = DataLoader(
            dataset,
            batch_size=spec['batch_size'],
            num_workers=8,
            pin_memory=True,
        )

        name = config_name.split('/')[-1].replace('.yaml', '')
        print('current dataset: {}'.format(name))

        scales = {'SRx2': 2, 'SRx3': 3, 'SRx4': 4}
        psnr, ssim = evaluate(
            loader,
            model_e,
            name=name,
            eval_y_only=config.get('eval_y_only'),
            eval_crop_size=config.get('eval_crop_size'),
            scale=scales[args.dataset] if args.dataset in scales.keys() else 1,
            ensemble=args.ensemble,
            save_dir=args.model.split('.pth')[0],
            verbose=True, 
            save_image=args.save,
            
        )

        print('result psnr : {:.4f} ssim : {:.4f}\n'.format(psnr, ssim))
