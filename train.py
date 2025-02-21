#File: train.py

import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # Ensure this is set before any other imports

import json
import warnings
import argparse
import random
import sys
import requests
import datetime
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from accelerate import Accelerator
from tqdm import tqdm
import wandb
from torchinfo import summary  # For detailed model summary

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
from models import create_model
from utils.utils import (
    seed_everything, load_checkpoint, save_checkpoints
)
from utils.loss import combined_loss

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




def log_model_details(model, log_file_path, config_dict=None, input_size=None):
    """
    Log detailed model architecture using torchinfo.summary along with parameter counts.
    If an input_size is provided, use torchinfo.summary for a detailed summary.
    Also log the training configuration if provided.
    """
    try:
        if input_size is None:
            # Fallback input size: (batch, channels, height, width)
            input_size = (1, 3, 256, 256)
        # Get the detailed model summary using torchinfo
        model_summary = summary(model, input_size=input_size, verbose=0)
        summary_str = str(model_summary)
    except Exception as e:
        print(f"Warning: torchinfo.summary failed with error: {e}. Using basic model string.")
        summary_str = str(model)
    
    # Also compute parameter counts mathematically
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    full_summary = (
        f"Model Detailed Summary:\n{'-'*50}\n"
        f"{summary_str}\n\n"
        f"Total Parameters: {param_count:,}\n"
        f"Trainable Parameters: {trainable_count:,}\n"
    )

    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(full_summary)
        if config_dict:
            f.write("\nTraining Configuration:\n")
            f.write(f"{'-'*50}\n")
            for key, value in config_dict.items():
                f.write(f"{key}: {value}\n")
    return full_summary


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
        
        table_rows.append([
            wandb.Image(inp_s.cpu().squeeze(0), caption=f"Input - {filename}"),
            wandb.Image(out_s.cpu().squeeze(0), caption=f"Prediction - {filename}"),
            wandb.Image(tar_s.cpu().squeeze(0), caption=f"Target - {filename}")
        ])
    
    validation_table = wandb.Table(
        columns=["Input", "Output", "Expected"],
        data=table_rows
    )
    
    wandb.log({
        "Epoch": epoch,
        "Validation_Images": validation_table
    })
def cfg_to_dict(cfg):
    """
    Recursively convert a YACS CfgNode (or any dict-like object) into a pure Python dictionary.
    """
    if not hasattr(cfg, "items"):
        return cfg
    out = {}
    for k, v in cfg.items():
        out[k] = cfg_to_dict(v)
    return out

def train():
    """
    Main training function:
      - Loads the YAML configuration.
      - Uses key-value overrides from the command line.
      - Initializes WandB logging (including the configuration and command line).
      - Logs a detailed model summary at the beginning.
      - Creates the model using the type specified in the config.
      - Contains resume logic for training from the latest checkpoint.
      - Saves checkpoints without redundant directory nesting.
    """
    # Simplified argument parser: only allow config file path and overrides.
    parser = argparse.ArgumentParser(description="Train Script for Underwater Image Enhancement using config file.")
    parser.add_argument("--config_yaml", type=str, default="config.yml", help="Path to the config YAML file.")
    parser.add_argument("--resume", action="store_true", help="Flag to resume training from the last checkpoint.")
    parser.add_argument("config_override", nargs="*", help="Optional config overrides as KEY VALUE pairs.")
    args = parser.parse_args()

    # Load and finalize the configuration.
    config = Config(args.config_yaml, args.config_override)
    config.finalize_config()
    # For convenience, work directly with the underlying CfgNode.
    opt = config._C

    # Create the log directory and log the command line and config.
    log_file_path = os.path.join(opt.LOG.LOG_DIR, opt.TRAINING.LOG_FILE)
    os.makedirs(opt.LOG.LOG_DIR, exist_ok=True)
    
    # Changed this part to properly handle file mode based on resume flag
    file_mode = 'a' if opt.TRAINING.RESUME else 'w'
    with open(log_file_path, file_mode, encoding='utf-8') as f:
        f.write("\n" + "="*50 + "\n")  # Add separator for better readability
        f.write(f"{'RESUMED' if opt.TRAINING.RESUME else 'NEW'} TRAINING SESSION: {datetime.datetime.now()}\n")
        f.write("Command line: " + " ".join(sys.argv) + "\n")
        f.write("Configuration:\n")
        f.write(str(opt))
        f.write("\n" + "="*50 + "\n")

    # 2) Reproducibility
    seed_everything(opt.OPTIM.SEED)

    # 3) Create model using the configuration (MODEL.NAME)
    try:
        model = create_model(opt)
    except ValueError as e:
        print(f"Error creating model: {e}")
        sys.exit(1)

    # 4) Initialize the accelerator and WandB.
    accelerator = Accelerator(log_with="wandb", mixed_precision=opt.TRAINING.MIXED_PRECISION)
    wandb.login(key='8d9ec67ac85ce634d875b480fed3604bfb9cb595')
    
    # Initialize run_id before wandb.init
    run_id = None
    
    # If resuming, try to get the wandb run ID from the checkpoint
    if opt.TRAINING.RESUME:
        ckp_dir = opt.TRAINING.SAVE_DIR
        last_ckpts = [f for f in os.listdir(ckp_dir) if f.startswith('last_checkpoint_epoch_') and f.endswith('.pth')]
        if last_ckpts:
            latest_ckpt = max(last_ckpts, key=lambda x: int(x.split('epoch_')[1].split('.')[0]))
            ckpt_path = os.path.join(ckp_dir, latest_ckpt)
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            run_id = checkpoint.get('wandb_run_id')
            if run_id:
                print(f"Resuming wandb run: {run_id}")
    
    if accelerator.is_local_main_process:
        try:
            # Create a combined name using dataset and session
            run_name = f"{opt.MODEL.DATASET_NAME}_{opt.MODEL.SESSION}"
            
            if is_online():
                print("Internet detected, using wandb online mode.")
                wandb.init(
                    project=f'{opt.MODEL.NAME}_underwater',
                    config=cfg_to_dict(opt),
                    name=run_name,
                    id=run_id,  # Use the loaded run_id
                    resume="allow"  # Changed from 'must' to 'allow'
                )
            else:
                print("No internet, using wandb offline mode.")
                wandb.init(
                    project=f'{opt.MODEL.NAME}_underwater',
                    config=cfg_to_dict(opt),
                    name=run_name,  # And here
                    mode='offline'
                )
        except wandb.errors.CommError:
            print("WandB initialization failed, using offline mode.")
            wandb.init(mode='offline')
        
        # NOTE: Do not update SAVE_DIR here because finalize_config() already appended WANDB.NAME.

        # Log additional information into WandB.
        wandb.config.update(
            {"command_line": " ".join(sys.argv)}, 
            allow_val_change=True  # Add this flag to allow config value changes
        )

        # Log the detailed model summary using an input size from config.
        input_size = (1, opt.MODEL.INPUT_CHANNELS, opt.TRAINING.PS_H, opt.TRAINING.PS_W)
        log_model_details(model, log_file_path, config_dict=cfg_to_dict(opt), input_size=input_size)

    device = accelerator.device

    # 5) Create optimizer and scheduler.
    criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

    # 6) Load data.
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

    # 7) Resume logic.
    start_epoch = 1
    best_psnr = 0.0
    best_ssim = 0.0
    best_loss = float("inf")
    best_psnr_epoch = 0
    best_ssim_epoch = 0
    best_loss_epoch = 0
    best_uciqe = 0.0
    best_uiqm = 0.0
    best_uciqe_epoch = 0
    best_uiqm_epoch = 0

    # Use the resume flag from the configuration (or command line override).
    if accelerator.is_local_main_process and opt.TRAINING.RESUME:
        ckp_dir = opt.TRAINING.SAVE_DIR
        ckpts = [f for f in os.listdir(ckp_dir) if f.endswith('.pth')]
        
        # Restrict to those that start with last_checkpoint_epoch_
        last_ckpts = [f for f in ckpts if f.startswith('last_checkpoint_epoch_')]
        if last_ckpts:
            def parse_epoch(fn):
                tokens = fn.split("epoch_")
                if len(tokens) < 2:
                    return -1
                try:
                    return int(tokens[-1].split(".pth")[0])
                except:
                    return -1

            latest_ckpt = max(last_ckpts, key=lambda x: parse_epoch(x))
            ckpt_path = os.path.join(ckp_dir, latest_ckpt)
            print(f"Resuming from {ckpt_path}")
            loaded_data = load_checkpoint(model, optimizer, scheduler, ckpt_path, device)
            run_id = loaded_data.get("wandb_run_id", None)
            start_epoch = loaded_data["epoch"] + 1
            best_psnr = loaded_data["best_psnr"]
            best_ssim = loaded_data["best_ssim"]
            best_loss = loaded_data["best_loss"]
            best_psnr_epoch = loaded_data["best_psnr_epoch"]
            best_ssim_epoch = loaded_data["best_ssim_epoch"]
            best_loss_epoch = loaded_data["best_loss_epoch"]
            best_metrics = loaded_data.get("best_metrics", {})
            best_psnr = best_metrics.get("best_psnr", best_psnr)
            best_ssim = best_metrics.get("best_ssim", best_ssim)
            print(f"Checkpoint loaded. Starting from epoch {start_epoch}")
        else:
            print("No 'last_checkpoint_epoch_' file found. Starting fresh.")

    # 8) Prepare with Accelerator.
    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader
    )

    # 9) Training loop.
    val_size = len(val_loader)
    start_time = time.time()  # Start timing the training process

    # Retrieve loss scales from the configuration.
    scale_psnr = opt.LOSSES.PSNR_SCALE
    scale_ssim = opt.LOSSES.SSIM_SCALE
    scale_lpips = opt.LOSSES.LPIPS_SCALE
    scale_edge = opt.LOSSES.EDGE_SCALE
    scale_freq = opt.LOSSES.FREQ_SCALE

    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        if accelerator.is_local_main_process:
            print(f"\n[Epoch {epoch}/{opt.OPTIM.NUM_EPOCHS}] Starting...")

        model.train()
        for i, data_batch in enumerate(tqdm(train_loader, disable=not accelerator.is_local_main_process)):
            inp, tar = data_batch[0], data_batch[1]
            inp, tar = inp.to(device), tar.to(device)

            optimizer.zero_grad()
            pred = model(inp)

            # Compute the combined loss with scaling factors from the config.
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

            # Gradient clipping based on config (if CLIP_GRAD is > 0).
            if opt.TRAINING.CLIP_GRAD > 0:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=opt.TRAINING.CLIP_GRAD)

            optimizer.step()

            if accelerator.is_local_main_process:
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

        # Validation after every VAL_AFTER_EVERY epochs.
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

                    out_val = model(inp_val)
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

                    out_g, tar_g = accelerator.gather((out_val, tar_val))

                    val_psnr += peak_signal_noise_ratio(out_g, tar_g, data_range=1).item()
                    val_ssim += structural_similarity_index_measure(out_g, tar_g, data_range=1).item()
                    val_lpips_loss += criterion_lpips(out_val, tar_val).item()
                   
                    val_uciqe += batch_uciqe(out_g)
                    val_uiqm += batch_uiqm(out_g)

            # Compute average metrics.
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

            # Update best metrics.
            if val_uciqe > best_uciqe:
                best_uciqe = val_uciqe
                best_uciqe_epoch = epoch
            if val_uiqm > best_uiqm:
                best_uiqm = val_uiqm
                best_uiqm_epoch = epoch

            metrics_dict = {
                "psnr": val_psnr,
                "ssim": val_ssim,
                "total_loss": val_total_loss,
                "best_psnr": best_psnr,
                "best_ssim": best_ssim,
                "best_loss": best_loss,
                "best_psnr_epoch": best_psnr_epoch,
                "best_ssim_epoch": best_ssim_epoch,
                "best_loss_epoch": best_loss_epoch,
                "best_uciqe": best_uciqe,
                "best_uiqm": best_uiqm,
                "best_uciqe_epoch": best_uciqe_epoch,
                "best_uiqm_epoch": best_uiqm_epoch
            }

            if accelerator.is_local_main_process:
                updated = save_checkpoints(model, optimizer, scheduler, epoch, metrics_dict, opt)
                best_psnr = updated["best_psnr"]
                best_ssim = updated["best_ssim"]
                best_loss = updated["best_loss"]
                best_psnr_epoch = updated["best_psnr_epoch"]
                best_ssim_epoch = updated["best_ssim_epoch"]
                best_loss_epoch = updated["best_loss_epoch"]

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
                
                wandb.log(val_metrics)
                log_validation_images(val_dataset, model, device, epoch)

                info_str = (
                    f"Epoch {epoch} || PSNR: {val_psnr:.4f} (best {best_psnr:.4f} @ {best_psnr_epoch}) | "
                    f"SSIM: {val_ssim:.4f} (best {best_ssim:.4f} @ {best_ssim_epoch}) | "
                    f"Loss: {val_total_loss:.4f} (best {best_loss:.4f} @ {best_loss_epoch}) | "
                    f"UCIQE: {val_uciqe:.4f} | UIQM: {val_uiqm:.4f}"
                )
                print(info_str)
                with open(log_file_path, mode='a', encoding='utf-8') as f:
                    f.write(json.dumps(info_str) + '\n')
                    wandb.save(log_file_path, base_path=opt.LOG.LOG_DIR)

    end_time = time.time()  # End timing the training process
    total_time = end_time - start_time
    if accelerator.is_local_main_process:
        final_metrics = (
            f"\nBest Metrics Summary:\n{'-'*50}\n"
            f"Best PSNR: {best_psnr:.4f} (epoch {best_psnr_epoch})\n"
            f"Best SSIM: {best_ssim:.4f} (epoch {best_ssim_epoch})\n"
            f"Best Loss: {best_loss:.4f} (epoch {best_loss_epoch})\n"
            f"Best UCIQE: {best_uciqe:.4f} (epoch {best_uciqe_epoch})\n"
            f"Best UIQM: {best_uiqm:.4f} (epoch {best_uiqm_epoch})\n"
            f"Total Training Time: {total_time:.2f} seconds\n"
        )

        with open(log_file_path, mode='a', encoding='utf-8') as f:
            f.write(final_metrics)

        wandb.log({
            "Best_Metrics/PSNR": best_psnr,
            "Best_Metrics/SSIM": best_ssim,
            "Best_Metrics/Loss": best_loss,
            "Best_Metrics/UCIQE": best_uciqe,
            "Best_Metrics/UIQM": best_uiqm,
            "Best_Metrics/PSNR_Epoch": best_psnr_epoch,
            "Best_Metrics/SSIM_Epoch": best_ssim_epoch,
            "Best_Metrics/Loss_Epoch": best_loss_epoch,
            "Best_Metrics/UCIQE_Epoch": best_uciqe_epoch,
            "Best_Metrics/UIQM_Epoch": best_uiqm_epoch,
            "Training/Total_Time": total_time
        })

        accelerator.end_training()

if __name__ == "__main__":
    train()