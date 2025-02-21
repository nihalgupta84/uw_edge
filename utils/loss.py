# File: utils/loss.py

import torch
import torch.nn.functional as F
from torch import nn
# Torchmetrics imports
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def edge_preservation_loss(predicted: torch.Tensor,
                           target: torch.Tensor,
                           device: str='cuda') -> torch.Tensor:
    """
    Use Sobel operators to compute difference in edges
    between predicted and target images.

    Returns a scalar L1 loss between their gradient magnitudes.
    """
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)

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
                  lambda_psnr: float = 1.0,
                  lambda_ssim: float = 0.3,
                  lambda_lpips: float = 0.7,
                  lambda_edge: float = 0.1,
                  lambda_freq: float = 0.05,
                  device: str = 'cuda') -> (torch.Tensor, dict):
    """
    Aggregates multiple sub-losses with user-defined scaling.
    """
    # Define or move these inside combined_loss if you prefer
    criterion_psnr = nn.SmoothL1Loss()
    criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)

    loss_psnr = criterion_psnr(predicted, target)
    # SSIM-based loss
    loss_ssim = 1.0 - structural_similarity_index_measure(predicted, target, data_range=1.0)
    # LPIPS-based loss
    loss_lpips = criterion_lpips(predicted, target)
    # Edge-based loss
    loss_edge = edge_preservation_loss(predicted, target, device=device)
    # Frequency-based loss
    loss_freq = frequency_domain_loss(predicted, target)

    total = (lambda_psnr * loss_psnr +
             lambda_ssim * loss_ssim +
             lambda_lpips * loss_lpips +
             lambda_edge * loss_edge +
             lambda_freq * loss_freq)

    losses_dict = {
        "Total_Loss": total.item(),
        "PSNR_Loss": loss_psnr.item(),
        "SSIM_Loss": loss_ssim.item(),
        "LPIPS_Loss": loss_lpips.item(),
        "Edge_Loss": loss_edge.item(),
        "Freq_Loss": loss_freq.item()
    }
    return total, losses_dict
