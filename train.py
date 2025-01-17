import json
import warnings
import os
import argparse
import time  # Added import for time module
import socket  # Added import for connectivity check

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
from models.edge_model import EdgeModel
from utils import *
import wandb
import requests

warnings.filterwarnings('ignore')
def is_online():
    try:
        # Attempt to connect to a reliable website
        response = requests.get('https://www.google.com', timeout=5)
        return True if response.status_code == 200 else False
    except requests.ConnectionError:
        return False
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
                project='edge_model',
                config=opt,
                name="t01",
                mode='online'
            )
        else:
            print("No internet, using wandb offline mode.")
            wandb.init(
                project='CCHRNET',
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
    model = EdgeModel()

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
    best_psnr = 0

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

            # forward
            optimizer_b.zero_grad()
            res = model(inp)

            loss_psnr = criterion_psnr(res, tar)
            loss_ssim = 1 - structural_similarity_index_measure(res, tar, data_range=1)
            loss_lpips = criterion_lpips(res, tar)

            train_loss = loss_psnr + 0.3 * loss_ssim + 0.7 * loss_lpips

            # backward
            accelerator.backward(train_loss)
            optimizer_b.step()

            # Log training metrics
            wandb.log({
                "Train Loss": train_loss.item(),
                "PSNR Loss": loss_psnr.item(),
                "SSIM Loss": loss_ssim.item(),
                "LPIPS Loss": loss_lpips.item(),
                "Learning Rate": scheduler_b.get_last_lr()[0],
                "Epoch": epoch,
                "Iteration": iteration
            })
            # iteration_end_time = time.time()
            # print(f"Iteration {iteration} took {iteration_end_time - iteration_start_time:.2f} seconds")


        scheduler_b.step()

        # testing
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()
            psnr = 0
            ssim = 0
            lpips = 0

            uciqe = 0
            uiqm = 0


            for _, data in enumerate(tqdm(testloader, disable=not accelerator.is_local_main_process)):
                # validation_start_time = time.time()
                inp = data[0].contiguous()
                tar = data[1]

                with torch.no_grad():
                    res = model(inp)

                res, tar = accelerator.gather((res, tar))

                psnr += peak_signal_noise_ratio(res, tar, data_range=1).item()
                ssim += structural_similarity_index_measure(res, tar, data_range=1).item()
                lpips += criterion_lpips(res, tar).item()
                uciqe += batch_uciqe(res)
                uiqm += batch_uiqm(res)

            psnr /= size
            ssim /= size
            lpips /= size
            uciqe /= size
            uiqm /= size

            # Log validation metrics
            wandb.log({
                "Validation PSNR": psnr,
                "Validation SSIM": ssim,
                "Validation LPIPS": lpips,
                "Validation UCIQE": uciqe,
                "Validation UIQM": uiqm,
                "Epoch": epoch
            })

            if psnr > best_psnr:
                # save model
                best_psnr = psnr
                best_epoch = epoch
                save_checkpoint({
                    'state_dict': model.state_dict(),
                }, epoch, opt.MODEL.SESSION, opt.TRAINING.SAVE_DIR)

            if accelerator.is_local_main_process:
                log_stats = ("epoch: {}, PSNR: {}, SSIM: {}, LPIPS: {}, UCIQE: {}, "
                             "UIQM: {}, best PSNR: {}, best epoch: {}"
                             .format(epoch, psnr, ssim, lpips, uciqe, uiqm, best_psnr, best_epoch))
                print(log_stats)
                log_file_path = os.path.join(log_dir, opt.TRAINING.LOG_FILE)
                # print(f"Writing log to: {log_file_path}")  # Debug print
                with open(log_file_path, mode='a', encoding='utf-8') as f:
                    f.write(json.dumps(log_stats) + '\n')
            
            validation_end_time = time.time()
            # print(f"Validation took {validation_end_time - validation_start_time:.2f} seconds")

        # epoch_end_time = time.time()
        # print(f"Epoch {epoch} took {epoch_end_time - epoch_start_time:.2f} seconds")

    accelerator.end_training()

if __name__ == '__main__':
    train()