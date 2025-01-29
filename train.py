import json
import warnings
import os
import argparse
import time  # Added import for time module
import socket  # Added import for connectivity check
import pywt

# Suppress Albumentations warnings
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import torch.optim as optim
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from config import Config
from data import get_data

from metrics.uciqe import batch_uciqe
from metrics.uiqm import batch_uiqm

from torchsampler import ImbalancedDatasetSampler

from models import *
from models.wavelet_model import WaveletModel
from utils import *
import wandb
import requests
import torch.nn.functional as F


warnings.filterwarnings('ignore')


def save_checkpoints(model, optimizer, scheduler, epoch, metrics, opt):
    """
    Saves different types of checkpoints based on individual metrics.
    
    This approach saves separate checkpoints for:
    1. Last checkpoint (for resuming training)
    2. Best PSNR checkpoint
    3. Best SSIM checkpoint 
    4. Best Loss checkpoint
    
    Args:
        model: Model instance
        optimizer: Optimizer instance
        scheduler: Scheduler instance
        epoch: Current epoch number
        metrics: Dictionary containing current and best metrics/losses
        opt: Configuration options
    """
    checkpoint_dir = os.path.join(opt.TRAINING.SAVE_DIR, opt.MODEL.SESSION)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Save last checkpoint for resuming training
    last_checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_psnr': metrics['best_psnr'],
        'best_ssim': metrics['best_ssim'],
        'best_loss': metrics['best_loss']
    }
    torch.save(last_checkpoint, 
               os.path.join(checkpoint_dir, 'last_checkpoint.pth'))

    # 2. Save best PSNR checkpoint if we have improvement
    if metrics['psnr'] > metrics['best_psnr']:
        metrics['best_psnr'] = metrics['psnr']
        metrics['best_psnr_epoch'] = epoch
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'psnr': metrics['psnr'],
            'other_metrics': {
                'ssim': metrics['ssim'],
                'total_loss': metrics['total_loss']
            }
        }, os.path.join(checkpoint_dir, 'best_psnr_model.pth'))
        print(f"Saved new best PSNR model: {metrics['psnr']:.4f}")

    # 3. Save best SSIM checkpoint if we have improvement
    if metrics['ssim'] > metrics['best_ssim']:
        metrics['best_ssim'] = metrics['ssim']
        metrics['best_ssim_epoch'] = epoch
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'ssim': metrics['ssim'],
            'other_metrics': {
                'psnr': metrics['psnr'],
                'total_loss': metrics['total_loss']
            }
        }, os.path.join(checkpoint_dir, 'best_ssim_model.pth'))
        print(f"Saved new best SSIM model: {metrics['ssim']:.4f}")

    # 4. Save best Loss checkpoint if we have improvement
    if metrics['total_loss'] < metrics['best_loss']:
        metrics['best_loss'] = metrics['total_loss']
        metrics['best_loss_epoch'] = epoch
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'total_loss': metrics['total_loss'],
            'other_metrics': {
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim']
            }
        }, os.path.join(checkpoint_dir, 'best_loss_model.pth'))
        print(f"Saved new best Loss model: {metrics['total_loss']:.4f}")

    # Remove any old epoch-specific checkpoints to save space
    old_checkpoints = [f for f in os.listdir(checkpoint_dir) 
                      if f.startswith('epoch_') and f.endswith('.pth')]
    for ckpt in old_checkpoints:
        os.remove(os.path.join(checkpoint_dir, ckpt))

    return metrics


def is_online():
    try:
        # Attempt to connect to a reliable website
        response = requests.get('https://www.google.com', timeout=5)
        return True if response.status_code == 200 else False
    except requests.ConnectionError:
        return False
    

def edge_preservation_loss(predicted, target, device='cuda'):
    """
    Computes edge preservation loss using Sobel operators
    
    Args:
        predicted: Tensor of predicted image (B,C,H,W), range [0,1]
        target: Tensor of ground truth image (B,C,H,W), range [0,1]
    
    Returns:
        Loss tensor (scalar)
    """
    # Define Sobel operators
    sobel_x = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32).to(device)
    sobel_y = torch.tensor([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=torch.float32).to(device)
    
    # Reshape Sobel operators for conv2d: (1,1,3,3)
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)
    
    # Process each channel separately
    loss = 0
    for c in range(predicted.shape[1]):
        # Extract channel: (B,H,W)
        pred_channel = predicted[:,c:c+1,:,:]  # Shape: (B,1,H,W)
        target_channel = target[:,c:c+1,:,:]
        
        # Apply Sobel operators
        pred_grad_x = F.conv2d(pred_channel, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_channel, sobel_y, padding=1)
        target_grad_x = F.conv2d(target_channel, sobel_x, padding=1)
        target_grad_y = F.conv2d(target_channel, sobel_y, padding=1)
        
        # Compute gradient magnitude
        pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-6)
        target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-6)
        
        # L1 loss between gradient magnitudes
        loss += F.l1_loss(pred_grad_mag, target_grad_mag)
    
    return loss / predicted.shape[1]  # Average over channels
def frequency_domain_loss(predicted: torch.Tensor, 
                              target: torch.Tensor) -> torch.Tensor:
    """
    Compute frequency-domain loss via 2D FFT magnitude comparison.
    
    Args:
        predicted (B,C,H,W): model output in [0,1]
        target    (B,C,H,W): ground truth in [0,1]
    Returns:
        torch.Tensor scalar: MSE loss on magnitude of the FFT.
    """
    # 1) Compute 2D FFT for predicted & target along last two dims (H,W)
    # Both remain on GPU.
    pred_fft = torch.fft.fft2(predicted, dim=(-2, -1))  
    targ_fft = torch.fft.fft2(target,    dim=(-2, -1))  

    # 2) Get magnitude
    pred_mag = torch.abs(pred_fft)
    targ_mag = torch.abs(targ_fft)

    # 3) Use MSE (or L1) on magnitudes
    freq_loss = F.mse_loss(pred_mag, targ_mag)
    
    return freq_loss

# def frequency_domain_loss(predicted, target, wavelet='haar', level=3, device='cuda'):
#     """
#     Computes wavelet coefficient loss for frequency domain analysis
    
#     Args:
#         predicted: Tensor of predicted image (B,C,H,W), range [0,1]
#         target: Tensor of ground truth image (B,C,H,W), range [0,1]
#         wavelet: Wavelet type to use
#         level: Number of wavelet decomposition levels
        
#     Returns:
#         Loss tensor (scalar)
#     """
#     # Process each image in batch separately
#     batch_size = predicted.shape[0]
#     total_loss = 0
    
#     for b in range(batch_size):
#         # Process each channel
#         channel_loss = 0
#         for c in range(predicted.shape[1]):
#             # Get single channel image
#             pred_img = predicted[b,c].cpu().numpy()
#             target_img = target[b,c].cpu().numpy()
            
#             # Compute wavelet coefficients
#             pred_coeffs = pywt.wavedec2(pred_img, wavelet, level=level)
#             target_coeffs = pywt.wavedec2(target_img, wavelet, level=level)
            
#             # Compare coefficients at each level
#             level_loss = 0
#             # First handle the approximation coefficients
#             level_loss += torch.nn.functional.mse_loss(
#                 torch.from_numpy(pred_coeffs[0]).to(device),
#                 torch.from_numpy(target_coeffs[0]).to(device)
#             )
            
#             # Then handle detail coefficients
#             for pred_detail, target_detail in zip(pred_coeffs[1:], target_coeffs[1:]):
#                 # Each detail tuple contains (horizontal, vertical, diagonal)
#                 for p_d, t_d in zip(pred_detail, target_detail):
#                     level_loss += torch.nn.functional.mse_loss(
#                         torch.from_numpy(p_d).to(device),
#                         torch.from_numpy(t_d).to(device)
#                     )
            
#             channel_loss += level_loss
            
#         total_loss += channel_loss / predicted.shape[1]  # Average over channels
        
#     return total_loss / batch_size  # Average over batch


def combined_loss(predicted, target, lambda_edge=0.1, lambda_freq=0.05, device='cuda'):
    """
    Combines all losses with appropriate weights and handles logging values
    
    Args:
        predicted: Predicted image tensor (B,C,H,W)
        target: Target image tensor (B,C,H,W)
        lambda_edge: Weight for edge loss (default: 0.1)
        lambda_freq: Weight for frequency loss (default: 0.05)
        device: Device to run computations on
    
    Returns:
        total_loss: Combined loss value
        loss_dict: Dictionary containing individual loss components for logging
    """
    # Base losses
    criterion_psnr = torch.nn.SmoothL1Loss()
    criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
    loss_psnr = criterion_psnr(predicted, target)
    loss_ssim = 1 - structural_similarity_index_measure(predicted, target, data_range=1.0)
    loss_lpips = criterion_lpips(predicted, target)
    
    # Additional losses
    loss_edge = edge_preservation_loss(predicted, target)
    loss_freq = frequency_domain_loss(predicted, target)
    
    # Combine losses
    total_loss = (loss_psnr + 
                 0.3 * loss_ssim + 
                 0.7 * loss_lpips +
                 lambda_edge * loss_edge +
                 lambda_freq * loss_freq)
    
    # Create dictionary of loss components for logging
    loss_dict = {
        'Total_Loss': total_loss.item(),
        'PSNR_Loss': loss_psnr.item(),
        'SSIM_Loss': loss_ssim.item(),
        'LPIPS_Loss': loss_lpips.item(),
        'Edge_Loss': loss_edge.item(),
        'Freq_Loss': loss_freq.item()
    }
    
    return total_loss, loss_dict

def train():
    # Accelerate
    opt = Config('config.yml')
    seed_everything(opt.OPTIM.SEED)

    # Initialize wandb with API key
    # wandb.login(key='8d9ec67ac85ce634d875b480fed3604bfb9cb595')  # Added wandb login
    # wandb_api_key = os.getenv('8d9ec67ac85ce634d875b480fed3604bfb9cb595')
    # if wandb_api_key:
    wandb.login(key='8d9ec67ac85ce634d875b480fed3604bfb9cb595')
    # else:
        # raise ValueError("WANDB_API_KEY environment variable not set")

    try:
        if is_online():
            print("Internet detected, using wandb online mode.")
            wandb.init(
                project='wavelet',
                config=opt,
                name="t01",
                mode='online'
            )
        else:
            print("No internet, using wandb offline mode.")
            wandb.init(
                project='wavelet',
                config=opt,
                name="t01",
                mode='offline'
            )
    except wandb.errors.CommError:
        print("WandB initialization failed, using offline mode.")
        wandb.init(mode='offline')
    accelerator = Accelerator(log_with='wandb') if opt.OPTIM.WANDB else Accelerator()
    if accelerator.is_local_main_process:
        os.makedirs(opt.TRAINING.SAVE_DIR, exist_ok=True)
        log_dir = os.path.abspath(opt.LOG.LOG_DIR)  # Ensure absolute path
        os.makedirs(log_dir, exist_ok=True)  # Ensure log directory exists
        # print(f"Log directory created or already exists: {log_dir}")  # Debug print
    device = accelerator.device

    config = {
        "dataset": opt.TRAINING.TRAIN_DIR
    }
    accelerator.init_trackers("UW", config=config)

    # Data Loader
    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir = opt.TRAINING.VAL_DIR

    print("Loading training data...")
    train_dataset = get_data(train_dir, opt.MODEL.INPUT, opt.MODEL.TARGET, 'train', opt.TRAINING.ORI,
                             {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    trainloader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=4,
                             drop_last=False, pin_memory=True)
    print("Training data loaded.")

    print("Loading validation data...")
    val_dataset = get_data(val_dir, opt.MODEL.INPUT, opt.MODEL.TARGET, 'test', opt.TRAINING.ORI,
                           {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True)
    print("Validation data loaded.")

    # Model & Loss
    model = WaveletModel()

    criterion_psnr = torch.nn.SmoothL1Loss()
    criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)

    # Optimizer & Scheduler
    optimizer_b = optim.AdamW(model.parameters(), lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999), eps=1e-8)
    scheduler_b = optim.lr_scheduler.CosineAnnealingLR(optimizer_b, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

    start_epoch = 1

    trainloader, testloader = accelerator.prepare(trainloader, testloader)
    model = accelerator.prepare(model)
    optimizer_b, scheduler_b = accelerator.prepare(optimizer_b, scheduler_b)

    best_epoch = 1
    # testing
    best_psnr = 0
    best_ssim = 0
    best_loss = float('inf')
    best_psnr_epoch = 0
    best_ssim_epoch = 0
    best_loss_epoch = 0

    size = len(testloader)

    # training
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        print(f"Starting epoch {epoch}...")
        # epoch_start_time = time.time()
        model.train()

        for iteration, data in enumerate(tqdm(trainloader, disable=not accelerator.is_local_main_process)):
            # iteration_start_time = time.time()
            inp = data[0].contiguous()
            tar = data[1]
            # Print the range of input and target images
            print(f"Input range: min={inp.min().item()}, max={inp.max().item()}")
            print(f"Target range: min={tar.min().item()}, max={tar.max().item()}")
            # forward
            optimizer_b.zero_grad()
            res = model(inp)

            # loss_psnr = criterion_psnr(res, tar)
            # loss_ssim = 1 - structural_similarity_index_measure(res, tar, data_range=1)
            # loss_lpips = criterion_lpips(res, tar)

            # train_loss = loss_psnr + 0.3 * loss_ssim + 0.7 * loss_lpips
            # Compute loss with new combined function
            train_losses, loss_components = combined_loss(res, tar, 
                                                    lambda_edge=0.1,  # Starting with conservative weights
                                                    lambda_freq=0.05,
                                                    device=device)            
            # Gradient clipping

            # backward
            accelerator.backward(train_losses)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer_b.step()

            if accelerator.is_local_main_process:
                wandb.log({
                    "Train/Total_Loss": loss_components['Total_Loss'],
                    "Train/PSNR_Loss": loss_components['PSNR_Loss'],
                    "Train/SSIM_Loss": loss_components['SSIM_Loss'],
                    "Train/LPIPS_Loss": loss_components['LPIPS_Loss'],
                    "Train/Edge_Loss": loss_components['Edge_Loss'],
                    "Train/Freq_Loss": loss_components['Freq_Loss'],
                    "Train/Learning_Rate": scheduler_b.get_last_lr()[0],
                    "Train/Epoch": epoch,
                    "Train/Iteration": iteration
                })
                    
            # iteration_end_time = time.time()
            # print(f"Iteration {iteration} took {iteration_end_time - iteration_start_time:.2f} seconds")


        scheduler_b.step()


        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()
            # Metrics (quality measurements)
            psnr = 0
            ssim = 0
            lpips = 0
            uciqe = 0
            uiqm = 0
            
            # Loss components
            total_loss = 0
            psnr_loss = 0
            ssim_loss = 0
            lpips_loss = 0
            edge_loss = 0
            freq_loss = 0

            for _, data in enumerate(tqdm(testloader, disable=not accelerator.is_local_main_process)):
                # validation_start_time = time.time()
                inp = data[0].contiguous()
                tar = data[1]

                with torch.no_grad():
                    res = model(inp)
                _, val_losses = combined_loss(res, tar, 
                                            lambda_edge=0.1, 
                                            lambda_freq=0.05,
                                            device=device)
                res, tar = accelerator.gather((res, tar))

                psnr += peak_signal_noise_ratio(res, tar, data_range=1).item()
                ssim += structural_similarity_index_measure(res, tar, data_range=1).item()
                lpips += criterion_lpips(res, tar).item()
                uciqe += batch_uciqe(res)
                uiqm += batch_uiqm(res)

                # Accumulate loss components
                total_loss += val_losses['Total_Loss']
                psnr_loss += val_losses['PSNR_Loss']
                ssim_loss += val_losses['SSIM_Loss']
                lpips_loss += val_losses['LPIPS_Loss']
                edge_loss += val_losses['Edge_Loss']
                freq_loss += val_losses['Freq_Loss']                
            size = len(testloader)
            psnr /= size
            ssim /= size
            lpips /= size
            uciqe /= size
            uiqm /= size
            # Average losses
            total_loss /= size
            psnr_loss /= size
            ssim_loss /= size
            lpips_loss /= size
            edge_loss /= size
            freq_loss /= size

            # Create metrics dictionary for checkpoint saving
            metrics = {
                'psnr': psnr,
                'ssim': ssim,
                'total_loss': total_loss,
                'best_psnr': best_psnr,
                'best_ssim': best_ssim,
                'best_loss': best_loss,
                'best_psnr_epoch': best_psnr_epoch,
                'best_ssim_epoch': best_ssim_epoch,
                'best_loss_epoch': best_loss_epoch
            }


            # Save checkpoints and update best metrics on main process only
            if accelerator.is_local_main_process:
                # Save checkpoints
                metrics = save_checkpoints(
                    model=model,
                    optimizer=optimizer_b,
                    scheduler=scheduler_b,
                    epoch=epoch,
                    metrics=metrics,
                    opt=opt
                )
                
                # Update our tracking variables
                best_psnr = metrics['best_psnr']
                best_ssim = metrics['best_ssim']
                best_loss = metrics['best_loss']
                best_psnr_epoch = metrics['best_psnr_epoch']
                best_ssim_epoch = metrics['best_ssim_epoch']
                best_loss_epoch = metrics['best_loss_epoch']

                # Log to wandb
                wandb.log({
                    # Quality Metrics
                    "Val/PSNR": psnr,
                    "Val/SSIM": ssim,
                    "Val/LPIPS": lpips,
                    "Val/UCIQE": uciqe,
                    "Val/UIQM": uiqm,
                    # Loss Components
                    "Val/Total_Loss": total_loss,
                    "Val/PSNR_Loss": psnr_loss,
                    "Val/SSIM_Loss": ssim_loss,
                    "Val/LPIPS_Loss": lpips_loss,
                    "Val/Edge_Loss": edge_loss,
                    "Val/Freq_Loss": freq_loss,
                    "Val/Epoch": epoch,
                    # Best Metrics
                    "Val/Best_PSNR": best_psnr,
                    "Val/Best_SSIM": best_ssim,
                    "Val/Best_Loss": best_loss
                })

                # Log to file
                log_stats = (f"epoch: {epoch}, "
                            f"PSNR: {psnr:.4f} (best: {best_psnr:.4f} @ epoch {best_psnr_epoch}), "
                            f"SSIM: {ssim:.4f} (best: {best_ssim:.4f} @ epoch {best_ssim_epoch}), "
                            f"Loss: {total_loss:.4f} (best: {best_loss:.4f} @ epoch {best_loss_epoch}), "
                            f"UCIQE: {uciqe:.4f}, UIQM: {uiqm:.4f}")
                
                print(log_stats)
                log_file_path = os.path.join(log_dir, opt.TRAINING.LOG_FILE)
                with open(log_file_path, mode='a', encoding='utf-8') as f:
                    f.write(json.dumps(log_stats) + '\n')
                print(f"Training completed after {opt.OPTIM.NUM_EPOCHS} epochs")
                wandb.log({"Training_Status": "Completed"})
    # accelerator.end_training()
if __name__ == '__main__':
    train()