#File: utils/utils.py

import math
import os
import random
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torchvision.utils import make_grid


def save_img(img, img_path, mode='RGB'):
    """
    Saves a tensor as an image using OpenCV.

    Args:
        img (torch.Tensor): Image tensor with shape (C, H, W) and values in [0, 1].
        img_path (str): Path to save the image.
        mode (str, optional): Color mode. Defaults to 'RGB'.
    """
    img = torch.squeeze(img)
    img = torch.transpose(img, 0, 1)
    img = torch.transpose(img, 1, 2).cpu().numpy() * 255
    if mode == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, img)


def seed_everything(seed=3407):
    """
    Seeds all relevant random number generators for reproducibility.

    Args:
        seed (int, optional): Seed value. Defaults to 3407.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, epoch, model_name, outdir):
    """
    Saves a checkpoint containing model, optimizer, and scheduler states.

    Args:
        state (dict): State dictionary containing model, optimizer, scheduler, and metrics.
        epoch (int): Current epoch number.
        model_name (str): Name of the model/session.
        outdir (str): Directory to save the checkpoint.
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    checkpoint_file = os.path.join(outdir, f"{model_name}_epoch_{epoch}.pth")
    torch.save(state, checkpoint_file)


def save_checkpoints(model, optimizer, scheduler, epoch, metrics, opt):
    """Saves various checkpoints: last, best PSNR, best SSIM, and best Loss."""
    checkpoint_dir = opt.TRAINING.SAVE_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Helper function to remove old checkpoints of a specific type
    def cleanup_old_checkpoints(prefix):
        old_ckpts = [f for f in os.listdir(checkpoint_dir) 
                     if f.startswith(prefix) and f.endswith('.pth')]
        for ckpt in old_ckpts:
            os.remove(os.path.join(checkpoint_dir, ckpt))

    # 1) Save and cleanup last checkpoint
    last_checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_psnr': metrics['best_psnr'],
        'best_ssim': metrics['best_ssim'],
        'best_loss': metrics['best_loss']
    }
    cleanup_old_checkpoints('last_checkpoint')
    save_checkpoint(last_checkpoint, epoch, 'last_checkpoint', checkpoint_dir)

    # 2) Save and cleanup best PSNR
    if metrics['psnr'] > metrics['best_psnr']:
        metrics['best_psnr'] = metrics['psnr']
        metrics['best_psnr_epoch'] = epoch
        best_psnr_checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'psnr': metrics['psnr'],
            'other_metrics': {
                'ssim': metrics['ssim'],
                'total_loss': metrics['total_loss']
            }
        }
        cleanup_old_checkpoints('best_psnr_model')
        save_checkpoint(best_psnr_checkpoint, epoch, 'best_psnr_model', checkpoint_dir)
        print(f"Saved new best PSNR model: {metrics['psnr']:.4f}")

    # 3) Save and cleanup best SSIM
    if metrics['ssim'] > metrics['best_ssim']:
        metrics['best_ssim'] = metrics['ssim']
        metrics['best_ssim_epoch'] = epoch
        best_ssim_checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'ssim': metrics['ssim'],
            'other_metrics': {
                'psnr': metrics['psnr'],
                'total_loss': metrics['total_loss']
            }
        }
        cleanup_old_checkpoints('best_ssim_model')
        save_checkpoint(best_ssim_checkpoint, epoch, 'best_ssim_model', checkpoint_dir)
        print(f"Saved new best SSIM model: {metrics['ssim']:.4f}")

    # 4) Save and cleanup best Loss
    if metrics['total_loss'] < metrics['best_loss']:
        metrics['best_loss'] = metrics['total_loss']
        metrics['best_loss_epoch'] = epoch
        best_loss_checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'total_loss': metrics['total_loss'],
            'other_metrics': {
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim']
            }
        }
        cleanup_old_checkpoints('best_loss_model')
        save_checkpoint(best_loss_checkpoint, epoch, 'best_loss_model', checkpoint_dir)
        print(f"Saved new best Loss model: {metrics['total_loss']:.4f}")

    # Remove any remaining epoch-specific checkpoints
    old_checkpoints = [f for f in os.listdir(checkpoint_dir) 
                       if f.startswith('epoch_') and f.endswith('.pth')]
    for ckpt in old_checkpoints:
        os.remove(os.path.join(checkpoint_dir, ckpt))

    return metrics


def load_checkpoint(model, optimizer, scheduler, weights, device):
    """
    Loads a checkpoint and updates the model, optimizer, and scheduler states.

    Args:
        model (torch.nn.Module): The model to load state into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to load state into.
        weights (str): Path to the checkpoint file.
        device (torch.device): Device to map the checkpoint.
    
    Returns:
        dict: A dictionary containing epoch, best_psnr, best_ssim, best_loss, 
              best_psnr_epoch, best_ssim_epoch, best_loss_epoch.
    """
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Checkpoint file {weights} not found.")

    checkpoint = torch.load(weights, map_location=device)
    
    # Handle DataParallel (possible 'module.' prefix in keys)
    new_state_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('module.'):
            new_key = key.replace('module.', '')
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict)
    
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    best_psnr = checkpoint.get('best_psnr', 0)
    best_ssim = checkpoint.get('best_ssim', 0)
    best_loss = checkpoint.get('best_loss', float('inf'))
    best_psnr_epoch = checkpoint.get('best_psnr_epoch', 0)
    best_ssim_epoch = checkpoint.get('best_ssim_epoch', 0)
    best_loss_epoch = checkpoint.get('best_loss_epoch', 0)
    
    return {
        'epoch': checkpoint.get('epoch', 1),
        'best_psnr': best_psnr,
        'best_ssim': best_ssim,
        'best_loss': best_loss,
        'best_psnr_epoch': best_psnr_epoch,
        'best_ssim_epoch': best_ssim_epoch,
        'best_loss_epoch': best_loss_epoch
    }
