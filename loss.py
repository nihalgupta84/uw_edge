import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure
)
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


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1) 
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4) 
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

class ContrastLoss(nn.Module):
    def __init__(self, ablation=False, loss_weight=1.0):

        super(ContrastLoss, self).__init__()

        self.vgg = Vgg19()
        if torch.cuda.is_available():
            self.vgg = self.vgg.cuda()
        # self.vgg = Vgg19()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation
        self.loss_weight = loss_weight


    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return self.loss_weight*loss