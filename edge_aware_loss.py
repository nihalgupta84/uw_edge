#FIle: edge_aware_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeAwareLoss(nn.Module):
    """
    A simple edge-aware loss that calculates the L1 difference
    between Sobel edges of pred and ref images.
    """
    def __init__(self, loss_weight=1.0):
        super(EdgeAwareLoss, self).__init__()
        self.loss_weight = loss_weight

        # Sobel kernels for X and Y directions
        # shape: (out_channels=1, in_channels=1, kernel_size=3,3)
        sobel_x = torch.FloatTensor([[-1,0,1],
                                     [-2,0,2],
                                     [-1,0,1]]).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.FloatTensor([[-1,-2,-1],
                                     [ 0, 0, 0],
                                     [ 1, 2, 1]]).unsqueeze(0).unsqueeze(0)

        # Register as buffers (non-trainable)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, ref):
        # pred, ref: (B, C, H, W), typically in [-1,1] or [0,1]
        # Convert to grayscale or pick one channel if you wish
        # For simplicity, we assume grayscale conversion by averaging channels
       
        device = pred.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)       
        pred_gray = torch.mean(pred, dim=1, keepdim=True)
        ref_gray  = torch.mean(ref,  dim=1, keepdim=True)

        # Apply Sobel in X, Y for pred
        edge_pred_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        edge_pred_y = F.conv2d(pred_gray, self.sobel_y, padding=1)
        edge_pred   = torch.sqrt(edge_pred_x**2 + edge_pred_y**2 + 1e-6)

        # Apply Sobel in X, Y for ref
        edge_ref_x = F.conv2d(ref_gray, self.sobel_x, padding=1)
        edge_ref_y = F.conv2d(ref_gray, self.sobel_y, padding=1)
        edge_ref   = torch.sqrt(edge_ref_x**2 + edge_ref_y**2 + 1e-6)

        # L1 difference of edges
        loss = F.l1_loss(edge_pred, edge_ref)

        return self.loss_weight * loss
