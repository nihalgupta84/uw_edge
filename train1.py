#File: train.py #version 3.0

import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # Ensure this is set before any other imports

# File: train.py

import json
import warnings
import argparse
import random
import sys
import requests
import datetime

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from accelerate import Accelerator
from tqdm import tqdm
import wandb
from torchinfo import summary  # Import torchinfo
import time  # Import time for timing the model

# Torchmetrics
from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Local Project Imports
from config import Config
from data import get_data
from metrics.uciqe import batch_uciqe
from metrics.uiqm import batch_uiqm
from models.wavelet_model_v2 import WaveletModel
from utils.utils1 import (
    seed_everything, load_checkpoint, save_checkpoints
)
from models.version_3 import Version3

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)  # Suppress UserWarnings


def is_online() -> bool:
    """
    Check for internet connectivity by pinging Google.
    Returns True if status code 200 is returned, else False.
    """
    try:
        response = requests.get("https://www.google.com", timeout=5)
        return (response.status_code == 200)
    except requests.ConnectionError:
        return False


def edge_preservation_loss(predicted: torch.Tensor,
                           target: torch.Tensor,
                           device: str='cuda') -> torch.Tensor:
    """
    Use Sobel operators to compute difference in edges
    between predicted and target images.

    Returns a scalar L1 loss between their gradient magnitudes.
    """
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                           dtype=torch.float32, device=device).view(1,1,3,3)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                           dtype=torch.float32, device=device).view(1,1,3,3)

    loss_val = 0.0
    for c in range(predicted.shape[1]):
        pred_ch = predicted[:, c:c+1, :, :]
        targ_ch = target[:, c:c+1, :, :]

        pred_grad_x = F.conv2d(pred_ch, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_ch, sobel_y, padding=1)
        targ_grad_x = F.conv2d(targ_ch, sobel_x, padding=1)
        targ_grad_y = F.conv2d(targ_ch, sobel_y, padding=1)

        pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-6)
        targ_grad_mag = torch.sqrt(targ_grad_x**2 + targ_grad_y**2 + 1e-6)
        loss_val += F.l1_loss(pred_grad_mag, targ_grad_mag)

    return loss_val / predicted.shape[1]


def frequency_domain_loss(predicted: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
    """
    Compare FFT magnitudes to encourage better frequency alignment.
    Returns MSE of magnitude difference.
    """
    pred_fft = torch.fft.fft2(predicted, dim=(-2, -1))
    targ_fft = torch.fft.fft2(target, dim=(-2, -1))

    pred_mag = torch.abs(pred_fft)
    targ_mag = torch.abs(targ_fft)

    return F.mse_loss(pred_mag, targ_mag)


def combined_loss(predicted: torch.Tensor,
                  target: torch.Tensor,
                  lambda_psnr: float=1.0,
                  lambda_ssim: float=0.3,
                  lambda_lpips: float=0.7,
                  lambda_edge: float=0.1,
                  lambda_freq: float=0.05,
                  device: str='cuda') -> (torch.Tensor, dict):
    """
    Aggregates multiple sub-losses with user-defined scaling.
    """
    criterion_psnr = nn.SmoothL1Loss()
    criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)


    loss_psnr = criterion_psnr(predicted, target)
    loss_ssim = 1.0 - structural_similarity_index_measure(predicted, target, data_range=1.0)
    loss_lpips = criterion_lpips(predicted, target)
    loss_edge = edge_preservation_loss(predicted, target, device=device)
    loss_freq = frequency_domain_loss(predicted, target)

    total = (lambda_psnr * loss_psnr
             + lambda_ssim * loss_ssim
             + lambda_lpips * loss_lpips
             + lambda_edge * loss_edge
             + lambda_freq * loss_freq)

    losses_dict = {
        "Total_Loss": total.item(),
        "PSNR_Loss": loss_psnr.item(),
        "SSIM_Loss": loss_ssim.item(),
        "LPIPS_Loss": loss_lpips.item(),
        "Edge_Loss": loss_edge.item(),
        "Freq_Loss": loss_freq.item()
    }
    return total, losses_dict


def log_model_details(model, log_file_path, wandb_config=None):
    """Log model architecture and parameters with proper Unicode handling"""
    try:
        # Generate model summary
        summary_str = str(model)
        param_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Format summary with parameter counts
        summary = (
            f"Model Architecture:\n{'-'*50}\n"
            f"{summary_str}\n\n"
            f"Total Parameters: {param_count:,}\n"
            f"Trainable Parameters: {trainable_count:,}\n"
        )

        # Write to file with UTF-8 encoding
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write(summary)
            
            # Log WandB config if provided
            if wandb_config:
                f.write("\nTraining Configuration:\n")
                f.write(f"{'-'*50}\n")
                for key, value in wandb_config.items():
                    f.write(f"{key}: {value}\n")
                    
        return summary

    except Exception as e:
        print(f"Warning: Error logging model details: {str(e)}")
        # Fallback to basic logging without special characters
        with open(log_file_path, 'w', encoding='ascii', errors='ignore') as f:
            f.write(f"Model Parameters: {param_count:,}\n")
            f.write(f"Trainable Parameters: {trainable_count:,}\n")


def log_validation_images(val_dataset, model, device, epoch, num_samples=2):
    """
    Log validation images to WandB in a separate section from metrics.
    """
    sample_indices = random.sample(range(len(val_dataset)), min(num_samples, len(val_dataset)))
    table_rows = []
    
    for idx in sample_indices:
        inp_s, tar_s, filename = val_dataset[idx]
        inp_s = inp_s.unsqueeze(0).to(device)
        tar_s = tar_s.unsqueeze(0).to(device)
        
        with torch.no_grad():
            out_s = model(inp_s)
        
        # Create a row with just the required columns in the correct order
        table_rows.append([
            wandb.Image(inp_s.cpu().squeeze(0), caption=f"Input - {filename}"),
            wandb.Image(out_s.cpu().squeeze(0), caption=f"Prediction - {filename}"),
            wandb.Image(tar_s.cpu().squeeze(0), caption=f"Target - {filename}")
        ])
    
    # Create table with explicit column names matching the data
    validation_table = wandb.Table(
        columns=["Input", "Output", "Expected"],
        data=table_rows
    )
    
    # Log the table
    wandb.log({
        "Epoch": epoch,
        "Validation_Images": validation_table
    })


def train():
    """
    Main Training Entry:
      * Loads config
      * Command-line overrides (both config + specific)
      * Mixed Precision, Gradient Clipping, custom WandB name
      * Accepts --resume for continuing training from last checkpoint
      * Logs everything (metrics, images, command-line) to WandB
      * Saves multiple checkpoint variants (last, best psnr/ssim/loss)
    """

    parser = argparse.ArgumentParser(description="Train Script with advanced CLI arguments and single checkpoint function.")
    parser.add_argument("--config_yaml", type=str, default="config.yml", help="Path to the config YAML file.")
    parser.add_argument("--resume", action="store_true", help="Flag to resume training from the last checkpoint.")
    parser.add_argument("config_override", nargs="*", help="Additional config overrides as KEY VALUE pairs.")

    # Overridable scaling factors
    parser.add_argument("--lambda_psnr", type=float, default=1.0, help="Scaling factor for PSNR-based SmoothL1 loss.")
    parser.add_argument("--lambda_ssim", type=float, default=0.3, help="Scaling factor for (1 - SSIM) loss.")
    parser.add_argument("--lambda_lpips", type=float, default=0.7, help="Scaling factor for LPIPS loss.")
    parser.add_argument("--lambda_edge", type=float, default=0.1, help="Scaling factor for edge preservation loss.")
    parser.add_argument("--lambda_freq", type=float, default=0.05, help="Scaling factor for frequency domain loss.")

    # Additional convenience arguments
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="Explicitly set the WandB run name. If provided, it will also set MODEL.SESSION.")
    parser.add_argument("--mixed_precision", type=str, default="no",
                        choices=["no", "fp16", "bf16"],
                        help="Choose mixed precision mode: 'no' (default), 'fp16', or 'bf16'.")
    parser.add_argument("--clip_grad", type=float, default=1.0,
                        help="Max norm for gradient clipping. Set to 0.0 or negative to disable.")

    args = parser.parse_args()
    config = Config(args.config_yaml, args.config_override)
 
    # If user supplies a wandb_name, override the config before finalize_config
    if args.wandb_name is not None:
        config._C.WANDB.NAME = args.wandb_name
        config._C.MODEL.SESSION = args.wandb_name  # Align MODEL.SESSION with WandB run name
    else:
        auto_run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config._C.WANDB.NAME = auto_run_name
        config._C.MODEL.SESSION = auto_run_name

    config.finalize_config()
    opt = config  # Short reference

    # Ensure log directory exists
    if not os.path.exists(opt.LOG.LOG_DIR):
        os.makedirs(opt.LOG.LOG_DIR)

    # 2) Reproducibility
    seed_everything(opt.OPTIM.SEED)

    # 3) Create model
    model = Version3()

    # Log model details
    log_file_path = os.path.join(opt.LOG.LOG_DIR, opt.TRAINING.LOG_FILE)

    # 4) Store user scaling factors
    scale_psnr = args.lambda_psnr
    scale_ssim = args.lambda_ssim
    scale_lpips = args.lambda_lpips
    scale_edge = args.lambda_edge
    scale_freq = args.lambda_freq

    # 5) Generate and log model summary
    accelerator = Accelerator(log_with="wandb", mixed_precision=args.mixed_precision)
    wandb.login(key='8d9ec67ac85ce634d875b480fed3604bfb9cb595')
    if accelerator.is_local_main_process:
        try:
            if is_online():
                print("Internet detected, using wandb online mode.")
                wandb.init(
                    project='wavelet_model_v2',
                    config=opt,
                    name=opt.WANDB.NAME,
                    resume='allow'
                )
            else:
                print("No internet, using wandb offline mode.")
                wandb.init(
                    project='wavelet_model_v2',
                    config=opt,
                    name=opt.WANDB.NAME,
                    mode='offline'
                )
        except wandb.errors.CommError:
            print("WandB initialization failed, using offline mode.")
            wandb.init(mode='offline')
        
        # Set the save directory to include the session name
        opt.TRAINING.SAVE_DIR = os.path.join(opt.TRAINING.SAVE_DIR, opt.MODEL.SESSION)
        os.makedirs(opt.TRAINING.SAVE_DIR, exist_ok=True)
        
        log_dir = os.path.abspath(opt.LOG.LOG_DIR)
        os.makedirs(log_dir, exist_ok=True)


        # Log the user-specified scaling factors
        wandb.config.update({
            "LossScale/PSNR": scale_psnr,
            "LossScale/SSIM": scale_ssim,
            "LossScale/LPIPS": scale_lpips,
            "LossScale/EDGE": scale_edge,
            "LossScale/FREQ": scale_freq
        })

        # Log the exact command line used to run this script
        wandb.config["command_line"] = " ".join(sys.argv)

        # Log the chosen mixed precision and gradient clip
        wandb.config["mixed_precision"] = args.mixed_precision
        wandb.config["clip_grad"] = args.clip_grad

        # Log model details after WandB init
# In train() function:
        log_model_details(
            model=model,
            log_file_path=log_file_path,
            wandb_config=config if accelerator.is_local_main_process else None
        )

    device = accelerator.device
    config_dict = {
        "dataset": opt.TRAINING.TRAIN_DIR
    }
    accelerator.init_trackers("UW", config=config_dict)


    # 8) Create optimizer, scheduler

    criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

    # 9) Load data
    print("Loading training data...")
    train_dataset = get_data(
        opt.TRAINING.TRAIN_DIR,
        opt.MODEL.INPUT,
        opt.MODEL.TARGET,
        mode='train',
        ori=opt.TRAINING.ORI,
        img_options={"w": opt.TRAINING.PS_W, "h": opt.TRAINING.PS_H}
    )
    train_loader = DataLoader(train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, 
                                num_workers=4, drop_last=False, pin_memory=True)
    print(f"Training samples: {len(train_dataset)}")

    print("Loading validation data...")
    val_dataset = get_data(
        opt.TRAINING.VAL_DIR,
        opt.MODEL.INPUT,
        opt.MODEL.TARGET,
        mode='test',
        ori=opt.TRAINING.ORI,
        img_options={"w": opt.TRAINING.PS_W, "h": opt.TRAINING.PS_H}
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                            num_workers=8, drop_last=False, pin_memory=True)
    print(f"Validation samples: {len(val_dataset)}")

    # 10) Possibly resume
    start_epoch = 1
    best_psnr = 0.0
    best_ssim = 0.0
    best_loss = float("inf")
    best_psnr_epoch = 0
    best_ssim_epoch = 0
    best_loss_epoch = 0

    if accelerator.is_local_main_process and opt.TRAINING.RESUME:
        ckp_dir = opt.TRAINING.SAVE_DIR
        ckpts = [f for f in os.listdir(ckp_dir) if f.endswith('.pth')]
        if ckpts:
            def parse_epoch(fn):
                tokens = fn.split("epoch_")
                if len(tokens) < 2:
                    return -1
                try:
                    return int(tokens[-1].split(".pth")[0])
                except:
                    return -1

            latest_ckpt = max(ckpts, key=lambda x: parse_epoch(x))
            ckpt_path = os.path.join(ckp_dir, latest_ckpt)
            print(f"Resuming from {ckpt_path}")
            loaded_data = load_checkpoint(model, optimizer, scheduler, ckpt_path, device)
            start_epoch = loaded_data["epoch"] + 1
            best_psnr = loaded_data["best_psnr"]
            best_ssim = loaded_data["best_ssim"]
            best_loss = loaded_data["best_loss"]
            best_psnr_epoch = loaded_data["best_psnr_epoch"]
            best_ssim_epoch = loaded_data["best_ssim_epoch"]
            best_loss_epoch = loaded_data["best_loss_epoch"]
            print(f"Checkpoint loaded. Starting from epoch {start_epoch}")
        else:
            print("No checkpoint found. Starting fresh.")

    # 11) Prepare with Accelerator
    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader
    )

    # 12) Start Training
    val_size = len(val_loader)
    start_time = time.time()  # Start timing the training process

    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        if accelerator.is_local_main_process:
            print(f"\n[Epoch {epoch}/{opt.OPTIM.NUM_EPOCHS}] Starting...")

        model.train()
        for i, data_batch in enumerate(tqdm(train_loader, disable=not accelerator.is_local_main_process)):
            inp, tar = data_batch[0], data_batch[1]
            inp, tar = inp.to(device), tar.to(device)

            optimizer.zero_grad()
            pred = model(inp)

            # Combined loss with custom scaling
            train_loss, loss_comp = combined_loss(
                pred, tar,
                lambda_psnr=scale_psnr,
                lambda_ssim=scale_ssim,
                lambda_lpips=scale_lpips,
                lambda_edge=scale_edge,
                lambda_freq=scale_freq,
                device=device
            )
            accelerator.backward(train_loss)

            # Gradient Clipping if user didn't set it to <=0
            if args.clip_grad > 0:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)

            optimizer.step()

            if accelerator.is_local_main_process:
                # Log training iteration-level metrics
                wandb.log({
                    "Train/Epoch": epoch,
                    "Train/Iter": i,
                    "Train/Total_Loss": loss_comp["Total_Loss"],
                    "Train/PSNR_Loss": loss_comp["PSNR_Loss"],
                    "Train/SSIM_Loss": loss_comp["SSIM_Loss"],
                    "Train/LPIPS_Loss": loss_comp["LPIPS_Loss"],
                    "Train/Edge_Loss": loss_comp["Edge_Loss"],
                    "Train/Freq_Loss": loss_comp["Freq_Loss"],
                    "Train/LR": scheduler.get_last_lr()[0],
                })

        scheduler.step()

        # Validation after every N epochs
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()
            val_psnr = 0.0
            val_ssim = 0.0
            val_lpips_sum = 0.0
            val_uciqe = 0.0
            val_uiqm = 0.0

            val_total_loss = 0.0
            val_psnr_loss = 0.0
            val_ssim_loss = 0.0
            val_lpips_loss = 0.0
            val_edge_loss = 0.0
            val_freq_loss = 0.0

            with torch.no_grad():
                for _, val_data in enumerate(tqdm(val_loader, disable=not accelerator.is_local_main_process)):
                    inp_val = val_data[0].contiguous()
                    tar_val = val_data[1]

                    with torch.no_grad():
                        out_val = model(inp_val)
                    # out_val, tar_val = accelerator.gather((out_val, tar_val))
                    # Combined loss
                    val_loss, val_dict = combined_loss(
                        out_val, tar_val,
                        lambda_psnr=scale_psnr,
                        lambda_ssim=scale_ssim,
                        lambda_lpips=scale_lpips,
                        lambda_edge=scale_edge,
                        lambda_freq=scale_freq,
                        device=device
                    )

                    val_total_loss += val_dict["Total_Loss"]
                    val_psnr_loss += val_dict["PSNR_Loss"]
                    val_ssim_loss += val_dict["SSIM_Loss"]
                    val_lpips_loss += val_dict["LPIPS_Loss"]
                    val_edge_loss += val_dict["Edge_Loss"]
                    val_freq_loss += val_dict["Freq_Loss"]

                    # Gather for metrics
                    out_g, tar_g = accelerator.gather((out_val, tar_val))

                    # Standard Metrics
                    val_psnr += peak_signal_noise_ratio(out_g, tar_g, data_range=1).item()
                    val_ssim += structural_similarity_index_measure(out_g, tar_g, data_range=1).item()
                    
                    val_lpips_loss += criterion_lpips(out_val, tar_val).item()

                   
                    val_uciqe += batch_uciqe(out_g)
                    val_uiqm += batch_uiqm(out_g)

                    # Debug prints to identify issues
                    # print(f"val_uciqe: {val_uciqe}, val_uiqm: {val_uiqm}")

            # Average
            val_total_loss /= val_size
            val_psnr_loss  /= val_size
            val_ssim_loss  /= val_size
            val_lpips_loss /= val_size
            val_edge_loss  /= val_size
            val_freq_loss  /= val_size

            val_psnr       /= val_size
            val_ssim       /= val_size
            val_lpips_sum  /= val_size
            val_uciqe      /= val_size
            val_uiqm       /= val_size

            # Prepare metrics dict for `save_checkpoints`
            metrics_dict = {
                "psnr": val_psnr,
                "ssim": val_ssim,
                "total_loss": val_total_loss,
                "best_psnr": best_psnr,
                "best_ssim": best_ssim,
                "best_loss": best_loss,
                "best_psnr_epoch": best_psnr_epoch,
                "best_ssim_epoch": best_ssim_epoch,
                "best_loss_epoch": best_loss_epoch
            }

            if accelerator.is_local_main_process:
                # Save checkpoints (last, best, etc.)
                updated = save_checkpoints(model, optimizer, scheduler, epoch, metrics_dict, opt)
                # Update best
                best_psnr = updated["best_psnr"]
                best_ssim = updated["best_ssim"]
                best_loss = updated["best_loss"]
                best_psnr_epoch = updated["best_psnr_epoch"]
                best_ssim_epoch = updated["best_ssim_epoch"]
                best_loss_epoch = updated["best_loss_epoch"]

                # Collect all metrics in a single dictionary
                val_metrics = {
                    "Val/Epoch": epoch,
                    "Val/PSNR": val_psnr,
                    "Val/SSIM": val_ssim,
                    "Val/LPIPS": val_lpips_sum,
                    "Val/UCIQE": val_uciqe,
                    "Val/UIQM": val_uiqm,
                    "Val/Total_Loss": val_total_loss,
                    "Val/PSNR_Loss": val_psnr_loss,
                    "Val/SSIM_Loss": val_ssim_loss,
                    "Val/LPIPS_Loss": val_lpips_loss,
                    "Val/Edge_Loss": val_edge_loss,
                    "Val/Freq_Loss": val_freq_loss,
                    "Val/Best_PSNR": best_psnr,
                    "Val/Best_SSIM": best_ssim,
                    "Val/Best_Loss": best_loss
                }
                
                # Log metrics
                wandb.log(val_metrics)
                
                # Log images separately
                log_validation_images(val_dataset, model, device, epoch)

                # Print for console
                info_str = (
                    f"Epoch {epoch} || PSNR: {val_psnr:.4f} (best {best_psnr:.4f} @ {best_psnr_epoch}) | "
                    f"SSIM: {val_ssim:.4f} (best {best_ssim:.4f} @ {best_ssim_epoch}) | "
                    f"Loss: {val_total_loss:.4f} (best {best_loss:.4f} @ {best_loss_epoch}) | "
                    f"UCIQE: {val_uciqe:.4f} | UIQM: {val_uiqm:.4f}"
                )
                print(info_str)
                log_file_path = os.path.join(opt.LOG.LOG_DIR, opt.TRAINING.LOG_FILE)
                with open(log_file_path,  mode='a', encoding='utf-8') as f:
                    f.write(json.dumps(info_str) + '\n')
                    wandb.save(log_file_path, base_path=opt.LOG.LOG_DIR)

    end_time = time.time()  # End timing the training process
    total_time = end_time - start_time
    # Log total training time
    with open(log_file_path, mode='a', encoding='utf-8') as f:
        f.write(f"Total Training Time: {total_time:.2f} seconds\n")
    wandb.log({"Total_Training_Time": total_time})

    accelerator.end_training()

if __name__ == "__main__":
    train()
