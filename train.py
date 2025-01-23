import argparse
import os
import random

import accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from tqdm import tqdm

import datasets
import models
import utils
from datasets.prefetch_dataloader import CUDAPrefetcher, CPUPrefetcher
from models.util.lr_scheduler import CosineAnnealingRestartCyclicLR
from test import evaluate

torch.cuda.empty_cache()

def print_only_once(thingy, timer=None):
    if accelerator.is_main_process:
        if timer:
            print(utils.time_text(timer.t()), end="  ")
            timer.s()
        print(thingy)

def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'SGD': SGD,
        'Adam': Adam,
        'AdamW': AdamW
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer

def make_lr_scheduler(optimizer, last_epoch, lr_scheduler_spec):
    LRScheduler = {
        'MultiStepLR': MultiStepLR,
        'CosineAnnealingLR': CosineAnnealingLR,
        'CosineAnnealingRestartCyclicLR': CosineAnnealingRestartCyclicLR
    }[lr_scheduler_spec['name']]
    lr_scheduler = LRScheduler(optimizer, last_epoch=last_epoch, **lr_scheduler_spec['args'])
    return lr_scheduler

def make_data_loader(spec, tag='', accelerator=None):
    if spec is None:
        return None

    batch_size = spec['batch_size']
    crop_size = spec['crop_size']
        
    dataset = datasets.make(spec['dataset'], {'crop_size': crop_size})
    num_data = len(dataset)

    if accelerator.is_main_process:
        log('{} dataset: size={}'.format(tag, num_data))
        for k, v in dataset[0].items():
            log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(tag == 'train'),
        num_workers=8,
        pin_memory=True,
        worker_init_fn=utils.numpy_init_dict[tag],
    )
    return loader, num_data, batch_size, crop_size

def make_data_loaders(accelerator):
    train_loader, num_data, batch_size, crop_size = make_data_loader(config.get('train_dataset'), tag='train', accelerator=accelerator)
    val_loader, _, _, _ = make_data_loader(config.get('valid_dataset'), tag='val', accelerator=accelerator)
    return train_loader, val_loader, num_data, batch_size, crop_size


def prepare_training(accelerator):
    device = accelerator.device

    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        state = sv_file['state']
        torch.set_rng_state(state)

        model, model_e = models.make(sv_file['model'], load_sd=True)
        model, model_e = model.to(device), model_e.to(device)

        optimizer = make_optimizer(model.parameters(), sv_file['optimizer'], load_sd=True)
        last_epoch = iter_start = sv_file['current_iter'] + 1

        print_only_once(f'Resuming from iter {iter_start}...')
    else:
        print_only_once('prepare_training from start')
        model, model_e = models.make(config['model'])
        model, model_e = model.to(device), model_e.to(device)

        model_params = dict(model.named_parameters())
        model_e_params = dict(model_e.named_parameters())
        for k in model_params.keys():
            model_e_params[k].data.mul_(0).add_(model_params[k].data, alpha=1)

        optimizer = make_optimizer(model.parameters(), config['optimizer'])
        last_epoch = -1
        iter_start = 1

    scheduler = make_lr_scheduler(optimizer, last_epoch, config['lr_scheduler'])

    model_e.eval()
    
    if accelerator.is_main_process:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
        log('model: #struct={}'.format(model))
    
    return model, model_e, optimizer, iter_start, scheduler

from torchvision import transforms
def train(accelerator, train_data, model, model_e, optimizer, scheduler, 
          batch_size, crop_size, cur_batch_size, cur_crop_size, use_grad_clip=True):

    model.train()
    loss_fn = nn.L1Loss()
    device = accelerator.device
    optimizer.zero_grad()

    # get the full sized lq and gt data
    lq = train_data['lq']
    gt = train_data['gt']

    # Progressive learning 
    # get partial random sample from lq and gt
    if cur_batch_size < batch_size:
        indices = random.sample(range(0, batch_size), k=cur_batch_size)
        lq = lq[indices]
        gt = gt[indices]

    # crop at random position of the max-cropped patch
    scaling_factor = int(gt.shape[-1] / lq.shape[-1])
    if cur_crop_size < crop_size:
        cur_crop_size = cur_crop_size // scaling_factor
        h_lr, w_lr = lq.shape[-2], lq.shape[-1]
        x0 = int((h_lr - cur_crop_size) * random.random())
        y0 = int((w_lr - cur_crop_size) * random.random())
        x1 = x0 + cur_crop_size
        y1 = y0 + cur_crop_size
        lq = lq[:, :, x0:x1, y0:y1]
        gt_x0 = int(x0 * scaling_factor)
        gt_y0 = int(y0 * scaling_factor)
        gt_x1 = int(x1 * scaling_factor)
        gt_y1 = int(y1 * scaling_factor)
        gt = gt[:, :, gt_x0:gt_x1, gt_y0:gt_y1]

    pred = model(lq)
    loss = loss_fn(pred, gt)

    # accumulate : seperate one batch into several steps to save mem
    # with accelerator.accumulate(model):
    optimizer.zero_grad()
    # loss.backward()
    accelerator.backward(loss)
    
    if use_grad_clip:
        clip_grad_norm_(model.parameters(), 0.01)   # same value used in restormer
    
    optimizer.step()
    scheduler.step()

    model_params = dict(model.named_parameters())
    model_e_params = dict(model_e.named_parameters())
    for k in model_params.keys():
        model_e_params[k].data.mul_(config['e_decay']).add_(model_params[k].data, alpha=1-config['e_decay'])

    return loss


def main(accelerator, config, log, writer, save_path):
    timer = utils.Timer()

    # data loaders
    train_loader, val_loader, num_train_data, batch_size, crop_size = make_data_loaders(accelerator=accelerator)
    model, model_e, optimizer, iter_start, scheduler = prepare_training(accelerator)
    model, model_e, optimizer, scheduler, train_loader = accelerator.prepare(
        model, model_e, optimizer, scheduler, train_loader
    )

    total_iter = config['total_iter']
    use_grad_clip = config['use_grad_clip']
    print_freq = config['print_freq']
    save_checkpoint_freq = config['save_checkpoint_freq']
    val_freq = config['val_freq']
    val_img_save_freq_mult = config.get('val_img_save_freq_mult', 300)
    save_last_freq = config['save_last_freq']
    
    max_val = -1e18
    
    prefetcher = CUDAPrefetcher(train_loader)
    print("ID: {} pre-setup took {}".format(accelerator.process_index, utils.time_text(timer.t())))

    # Progressive learning 
    progressive_train = config.get('progressive_train')
    if progressive_train:
        progressive_iters = progressive_train.get('iters')
        batch_sizes = progressive_train.get('batch_sizes')
        crop_sizes = progressive_train.get('crop_sizes')
        assert batch_size >= max(batch_sizes), f"dataset batch size {batch_size} smaller than batch sizes {batch_sizes} for progressive trainning"
        assert crop_size >= max(crop_sizes), f"dataset crop size {crop_size} smaller than crop sizes {crop_sizes} for progressive trainning"
        # sum the milestone up
        progressive_learning_trigger = []
        if progressive_iters:
            progressive_iters = np.array([sum(progressive_iters[0:i + 1]) for i in range(0, len(progressive_iters))])
            progressive_learning_trigger = [True] * len(progressive_iters)


    current_iter = iter_start - 1
    # epoch count will be wrong if iter passed any of the PL milestone
    # but since this is only used in setting seed, ignore it for now
    epoch = current_iter // (num_train_data // batch_size)  

    with tqdm(initial=current_iter, total=total_iter, disable=True) as pbar:
        while current_iter <= total_iter:
            # use epoch as seed, this func only affect the iter(dataloader)
            # as it will reset to original seed after the creation      
            prefetcher.reset(epoch+999)     
            train_data = prefetcher.next()
    
            while train_data:
                current_iter += 1
                if current_iter > total_iter:
                    break   

                # log loop info
                if accelerator.is_main_process:
                    log_info = []
                    t_iter_start = timer.t()
                    if current_iter % print_freq == 0:
                        log_info = ['iter {}/{}'.format(current_iter, total_iter)]
                        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], current_iter)
                        log_info.append('lr:{}'.format(optimizer.param_groups[0]['lr']))

                if progressive_train:
                    bs_j = np.argmax(current_iter <= progressive_iters)
                    # get the batch_size and patch_size for current step
                    cur_batch_size = batch_sizes[bs_j]
                    cur_crop_size = crop_sizes[bs_j]
                    if progressive_learning_trigger[bs_j]:
                        # there will (hopefully) be gpu_count of pbars, so "\n"*gpu_count
                        gpu_count = torch.cuda.device_count()
                        print_only_once('\nUpdating Patch_Size to {} and Batch_Size to {} ({}*{})\n'.format(
                            cur_crop_size, cur_batch_size*gpu_count, cur_batch_size, gpu_count))
                        progressive_learning_trigger[bs_j] = False
                else:
                    cur_batch_size = batch_size # use full batch
                    cur_crop_size = crop_size   # never crop
                    
                train_loss = train(accelerator, train_data, model, model_e, optimizer, scheduler, 
                                batch_size, crop_size, cur_batch_size, cur_crop_size, use_grad_clip=use_grad_clip)
                train_data = prefetcher.next()
                # barrier 
                accelerator.wait_for_everyone()
                # update progress bar
                pbar.update(1)

                # saving / printing / etc..
                if accelerator.is_main_process:
                    # log loss info
                    if current_iter % print_freq == 0:
                        writer.add_scalars('loss', {'train': train_loss}, current_iter)
                        log_info.append('train: loss={:.4f}'.format(train_loss))
                    # skip unwrap if no need
                    if current_iter % save_last_freq != 0 and current_iter % save_checkpoint_freq != 0 and \
                        current_iter % val_freq != 0:
                        continue

                    unwrap_model = accelerator.unwrap_model(model)
                    unwrap_model_e = accelerator.unwrap_model(model_e)

                    model_spec = config['model']
                    model_spec['sd'] = unwrap_model.state_dict()
                    model_spec['sd_e'] = unwrap_model_e.state_dict()
                    optimizer_spec = config['optimizer']
                    optimizer_spec['sd'] = optimizer.state_dict()

                    # add state to sv_file
                    state = torch.get_rng_state()
                    sv_file = {'model': model_spec, 'optimizer': optimizer_spec, 'current_iter': current_iter, 'state': state}
                    
                    # save last
                    if current_iter == total_iter or ((save_last_freq is not None) and (current_iter % save_last_freq == 0)):
                        torch.save(sv_file, os.path.join(save_path, 'current_iter-last.pth'))

                    # save ckpt
                    if (save_checkpoint_freq is not None) and (current_iter % save_checkpoint_freq == 0):
                        torch.save(sv_file, os.path.join(save_path, 'current_iter-{}.pth'.format(current_iter)))

                    # eval
                    with torch.no_grad():
                        if (val_freq is not None) and (current_iter % val_freq == 0):
                            psnr, ssim = evaluate(
                                val_loader,
                                unwrap_model_e,
                                name="validation",
                                eval_y_only=config.get('eval_y_only'),
                                eval_crop_size=config.get('eval_crop_size'),
                                save_dir=save_path 
                                    if ((val_img_save_freq_mult is not None) and (current_iter % (val_freq*val_img_save_freq_mult) == 0))
                                    else None,      # save only on multiple of val_freq
                                current_iter=current_iter
                            )

                            log_val_msg = 'val: psnr={:.4f} ssim={:.4f}'.format(psnr, ssim)
                            writer.add_scalars('psnr', {'val': psnr}, current_iter)
                            writer.add_scalars('ssim', {'val': ssim}, current_iter)

                            log_info.append(log_val_msg)

                            if psnr > max_val:
                                max_val = psnr
                                torch.save(sv_file, os.path.join(save_path, 'current_iter-best.pth'))

                    t = timer.t()
                    prog = (current_iter - iter_start + 1) / (total_iter - iter_start + 1)
                    t_iter = utils.time_text(t - t_iter_start)
                    t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                    log_info.append('{} {}/{}'.format(t_iter, t_elapsed, t_all))

                    log(', '.join(log_info))
                    writer.flush()
                    
            epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--find_unused_parameters', action='store_true', default=False)
    parser.add_argument('--split_batches', action='store_true', default=False)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    args = parser.parse_args()
    
    dataloader_config = accelerate.DataLoaderConfiguration(split_batches=args.split_batches)
    # split batch : 
    #   False : len of datasets will be divided 
    #   (since batch will be multiplied)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
    accelerator = Accelerator(
        dataloader_config=dataloader_config,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        step_scheduler_with_optimizer=False,    # step once every iter, not when every GPU step
        kwargs_handlers=[ddp_kwargs],
    )

    utils.set_seed(2454)
    accelerate.utils.set_seed(2454, device_specific=True)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print_only_once('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
        
    save_path = os.path.join('save', save_name)

    log, writer = None, None
    if accelerator.is_main_process:
        log, writer, save_path = utils.set_save_path(save_path)
        with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, sort_keys=False)

    main(accelerator, config, log, writer, save_path)
