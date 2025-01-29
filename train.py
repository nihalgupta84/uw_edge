import json
import warnings
import os

import time  # Added import for time module

from edge_aware_loss import EdgeAwareLoss  # Added import for EdgeAwareLoss class
# Suppress Albumentations warnings
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import argparse
import torch.optim as optim
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

# from config import Config
from data import get_data
from config.config import Config
from metrics.uciqe import batch_uciqe
from metrics.uiqm import batch_uiqm

from torchsampler import ImbalancedDatasetSampler

from models import *
from models.edge_model import EdgeModel
from utils import *
import wandb
import requests
import random
from torchvision.utils import save_image

warnings.filterwarnings('ignore')
def is_online():
    try:
        # Attempt to connect to a reliable website
        response = requests.get('https://www.google.com', timeout=5)
        return True if response.status_code == 200 else False
    except requests.ConnectionError:
        return False
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str, default='config.yml')
    parser.add_argument('--resume', action='store_true')  # Change to action flag
    parser.add_argument('config_override', nargs='*')
    args = parser.parse_args()
    config = Config(args.config_yaml, args.config_override)
    config.finalize_config()
    
    opt = config  # Use the instantiated config with overrides
    
    seed_everything(opt.OPTIM.SEED)

    # Initialize wandb with API key
    wandb.login(key='8d9ec67ac85ce634d875b480fed3604bfb9cb595')

    accelerator = Accelerator(log_with='wandb') if opt.OPTIM.WANDB else Accelerator()

    model = EdgeModel()
    if accelerator.is_local_main_process:
        try:
            if is_online():
                print("Internet detected, using wandb online mode.")
                wandb.init(
                    project='edge_model',
                    config=opt,
                    name=opt.WANDB.NAME,
                    resume='allow'
                )
            else:
                print("No internet, using wandb offline mode.")
                wandb.init(
                    project='CCHRNET',
                    config=opt,
                    name=opt.WANDB.NAME,
                    mode='offline'
                )
        except wandb.errors.CommError:
            print("WandB initialization failed, using offline mode.")
            wandb.init(mode='offline')
        
        os.makedirs(opt.TRAINING.SAVE_DIR, exist_ok=True)
        log_dir = os.path.abspath(opt.LOG.LOG_DIR)
        os.makedirs(log_dir, exist_ok=True)

    device = accelerator.device

    config_dict = {
        "dataset": opt.TRAINING.TRAIN_DIR
    }
    accelerator.init_trackers("UW", config=config_dict)

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
    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=16, drop_last=False,
                            pin_memory=True)
    print("Validation data loaded.")

    # Model & Loss
    # model = EdgeModel()  # Move model definition before loading checkpoints

    criterion_psnr = torch.nn.SmoothL1Loss()
    criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
    edge_criterion = EdgeAwareLoss(loss_weight=0.1)  # Initialize EdgeAwareLoss

    # Optimizer & Scheduler
    optimizer_b = optim.AdamW(model.parameters(), lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999), eps=1e-8)
    scheduler_b = optim.lr_scheduler.CosineAnnealingLR(optimizer_b, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

    trainloader, testloader, model, optimizer_b, scheduler_b = accelerator.prepare(trainloader, testloader, model, optimizer_b, scheduler_b)

    best_epoch = 1
    best_psnr = 0

    if accelerator.is_local_main_process and opt.TRAINING.RESUME:
        # Find the latest checkpoint
        checkpoints = [f for f in os.listdir(opt.TRAINING.SAVE_DIR) if f.endswith('.pth')]
        if checkpoints:
            latest_checkpoint = max(
                checkpoints,
                key=lambda x: int(x.split('_epoch_')[-1].split('.pth')[0])
            )
            checkpoint_path = os.path.join(opt.TRAINING.SAVE_DIR, latest_checkpoint)
            start_epoch, best_psnr, best_epoch = load_checkpoint(model, optimizer_b, scheduler_b, checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}, starting from epoch {start_epoch + 1}")
            start_epoch += 1
        else:
            print("No checkpoints found, starting from epoch 1.")
            start_epoch = 1
    else:
        start_epoch = 1

    size = len(testloader)

    # training
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        if accelerator.is_local_main_process:
            print(f"Starting epoch {epoch}...")
        model.train()

        for iteration, data in enumerate(tqdm(trainloader, disable=not accelerator.is_local_main_process)):
            inp = data[0].contiguous()
            tar = data[1]

            optimizer_b.zero_grad()
            res = model(inp)

            loss_psnr = criterion_psnr(res, tar)
            loss_ssim = 1 - structural_similarity_index_measure(res, tar, data_range=1)
            loss_lpips = criterion_lpips(res, tar)
            loss_edge = edge_criterion(res, tar)  # Compute edge-aware loss

            # train_loss = loss_psnr + 0.3 * loss_ssim + 0.7 * loss_lpips + loss_edge  # Add edge-aware loss
            loss_val = 0.0
            if opt.LOSSES.USE_PSNR:
                loss_val += opt.LOSSES.PSNR_SCALE * loss_psnr
            if opt.LOSSES.USE_SSIM:
                loss_val += opt.LOSSES.SSIM_SCALE * loss_ssim
            if opt.LOSSES.USE_LPIPS:
                loss_val += opt.LOSSES.LPIPS_SCALE * loss_lpips
            if opt.LOSSES.USE_EDGE:
                loss_val += opt.LOSSES.EDGE_SCALE * loss_edge

            train_loss = loss_val
            # backward
            accelerator.backward(train_loss)
            optimizer_b.step()

            if accelerator.is_local_main_process:
                # Log training metrics
                wandb.log({
                    "Train Loss": train_loss.item(),
                    "PSNR Loss": loss_psnr.item(),
                    "SSIM Loss": loss_ssim.item(),
                    "LPIPS Loss": loss_lpips.item(),
                    "Edge Loss": loss_edge.item(),
                    'scales/psnr_scale': opt.LOSSES.PSNR_SCALE,
                    'scales/ssim_scale': opt.LOSSES.SSIM_SCALE,
                    'scales/lpips_scale': opt.LOSSES.LPIPS_SCALE,
                    'scales/edge_scale': opt.LOSSES.EDGE_SCALE,
                    "Learning Rate": scheduler_b.get_last_lr()[0],
                    "Epoch": epoch,
                    "Iteration": iteration
                })

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
            if accelerator.is_local_main_process:
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
                        'optimizer': optimizer_b.state_dict(),
                        'scheduler': scheduler_b.state_dict(),
                        'best_psnr': best_psnr,
                        'best_epoch': best_epoch
                    }, epoch, opt.WANDB.NAME, opt.TRAINING.SAVE_DIR)

                log_stats = ("epoch: {}, PSNR: {}, SSIM: {}, LPIPS: {}, UCIQE: {}, "
                             "UIQM: {}, best PSNR: {}, best epoch: {}"
                             .format(epoch, psnr, ssim, lpips, uciqe, uiqm, best_psnr, best_epoch))
                print(log_stats)
                log_file_path = os.path.join(log_dir, opt.TRAINING.LOG_FILE)
                with open(log_file_path, mode='a', encoding='utf-8') as f:
                    f.write(json.dumps(log_stats) + '\n')
                    wandb.save(log_file_path, base_path=log_dir)
                
                # Save and log random sample images
                num_samples = 2
                sample_indices = random.sample(range(len(testloader.dataset)), num_samples)

                table = wandb.Table(columns=["Input", "Target", "Prediction"])
                for idx in sample_indices:
                    inp, tar, filename = testloader.dataset[idx]
                    inp = inp.unsqueeze(0).to(device)
                    tar = tar.unsqueeze(0).to(device)
                    with torch.no_grad():
                        res = model(inp)
                    # print(f"Input shape: {inp.cpu().squeeze(0).shape}")
                    # print(f"Target shape: {tar.cpu().squeeze(0).shape}")
                    # print(f"Prediction shape: {res.cpu().squeeze(0).shape}")
                    # Comment out local saving:
                    # input_path = os.path.join(log_dir, f"input_epoch_{epoch}_{filename}")
                    # target_path = os.path.join(log_dir, f"target_epoch_{epoch}_{filename}")
                    # res_path = os.path.join(log_dir, f"res_epoch_{epoch}_{filename}")
                    # save_image(inp, input_path)
                    # save_image(tar, target_path)
                    # save_image(res, res_path)

                    # Add to wandb table without batch dimension:
                    table.add_data(
                        wandb.Image(inp.cpu().squeeze(0), caption=f"Input Epoch {epoch} - {filename}"),
                        wandb.Image(tar.cpu().squeeze(0), caption=f"Target Epoch {epoch} - {filename}"),
                        wandb.Image(res.cpu().squeeze(0), caption=f"Prediction Epoch {epoch} - {filename}")
                    )

                wandb.log({"Samples": table})

    accelerator.end_training()


if __name__ == '__main__':
    main()