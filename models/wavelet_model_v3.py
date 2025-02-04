#File: models/version_3.py

##############################################################################
# FinalUnifiedUWModel.py
# FinalUnifiedUWModel integrates:
#  • Wavelet+Color Prior (with reflection padding)
#  • SC Edge Detection and Channel-Spatial Attention in encoder/decoder blocks
#  • A Quaternion Cross-Attention Block at the bottleneck
#  • A learned Scale Harmonizer at the output
#
# This complete implementation is designed to robustly enhance underwater images
# and beat the state-of-the-art by addressing global color distortions, preserving
# local details, and avoiding known artifact issues.
##############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
try:
    from pytorch_wavelets import DWTForward
except ImportError:
    raise ImportError("Install pytorch_wavelets for wavelet decomposition.")

# ---------------------------------------------------------------------------
# 1. Wavelet+Color Prior Module
# -----------------------------------
# 
#----------------------------------------

def check_nan(x, msg):
    """Utility to check for NaNs and print a message. 
       You can comment this out if everything is stable."""
    if torch.isnan(x).any():
        print(f"[NaN Alert] in {msg}")
        print(f"mean and standard deviation : {x.mean()}, {x.std()}")
    return x


class SimpleWaveletColorPrior(nn.Module):
    """
    Performs a single-level wavelet transform (using reflection padding)
    to extract the LL band. A small color-shift network is applied to the
    original image. The upsampled LL band, the original image, and the
    color-shifted image are concatenated to form a 9-channel fused input.
    """
    def __init__(self, wave='haar'):
        super(SimpleWaveletColorPrior, self).__init__()
        # Use reflection padding to avoid border artifacts
        self.dwt = DWTForward(J=1, wave=wave, mode='reflect')
        self.color_shift = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=1, bias=True)
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        B, C, H, W = x.shape
        Yl, Yh = self.dwt(x)  # Yl: (B,3,H/2,W/2)
        LL = Yl
        color_map = self.color_shift(x)
        LL_up = F.interpolate(LL, size=(H, W), mode='bilinear', align_corners=False)
        # Concatenate along channel dimension: 3 (x) + 3 (color_map) + 3 (LL_up) = 9 channels
        fused = torch.cat([x, color_map, LL_up], dim=1)
        return fused

# ---------------------------------------------------------------------------
# 2. SC Edge Detection and Attention Modules
# ---------------------------------------------------------------------------
class SCEdgeDetectionModule(nn.Module):
    """
    Implements depthwise convolution‐based edge detection.
    It uses three kernels:
      - Center-difference (custom designed)
      - Sobel horizontal
      - Sobel vertical
    Their outputs are concatenated and fused to provide robust edge features.
    """
    def __init__(self, channels):
        super(SCEdgeDetectionModule, self).__init__()
        self.cdc = nn.utils.weight_norm(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        )
        self.hdc = nn.utils.weight_norm(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        )
        self.vdc = nn.utils.weight_norm(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.InstanceNorm2d(channels),
            nn.GELU()
        )
        self._init_edge_kernels()

    def _init_edge_kernels(self):
        # Center-difference kernel
        cdc_kernel = torch.zeros(1, 1, 3, 3)
        cdc_kernel[0, 0, 1, 1] = 1
        cdc_kernel[0, 0, :, :] -= 1/8
        cdc_kernel = cdc_kernel / (cdc_kernel.abs().sum() + 1e-5)
        # Sobel horizontal kernel
        hdc_kernel = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]]).float().view(1, 1, 3, 3) / 8.0
        # Sobel vertical kernel
        vdc_kernel = torch.tensor([[-1, -2, -1],
                                   [ 0,  0,  0],
                                   [ 1,  2,  1]]).float().view(1, 1, 3, 3) / 8.0
        self.register_buffer('cdc_kernel', cdc_kernel)
        self.register_buffer('hdc_kernel', hdc_kernel)
        self.register_buffer('vdc_kernel', vdc_kernel)
        self.cdc.weight.data = cdc_kernel.repeat(self.cdc.weight.shape[0], 1, 1, 1)
        self.hdc.weight.data = hdc_kernel.repeat(self.hdc.weight.shape[0], 1, 1, 1)
        self.vdc.weight.data = vdc_kernel.repeat(self.vdc.weight.shape[0], 1, 1, 1)

    def forward(self, x):
        cdc_out = self.cdc(x)
        hdc_out = self.hdc(x)
        vdc_out = self.vdc(x)
        fused = torch.cat([cdc_out, hdc_out, vdc_out], dim=1)
        return self.fusion(fused)

class SCAttention(nn.Module):
    """
    Combines channel attention and spatial attention along with an edge-enhancement branch.
    This module recalibrates features using both global (channel) and local (spatial) cues.
    """
    def __init__(self, channels, reduction=8):
        super(SCAttention, self).__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.LayerNorm([channels, 1, 1]),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1)
        )
        self.layer_norm = nn.LayerNorm([channels, 1, 1])
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.InstanceNorm2d(channels),
            nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.InstanceNorm2d(channels),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # Channel branch
        c_att = self.channel_gate(x)
        c_att = self.layer_norm(c_att)
        ch_out = x * c_att.sigmoid()
        # Spatial branch
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.amax(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sp_att = self.spatial_gate(spatial_input)
        sp_out = x * sp_att
        # Edge branch
        edge_out = self.edge_conv(x)
        # Fusion of channel and spatial
        combined = torch.cat([ch_out, sp_out], dim=1)
        return self.fusion(combined) + edge_out

# ---------------------------------------------------------------------------
# 3. SC Encoder and Decoder Blocks
# ---------------------------------------------------------------------------
class SCEncoderBlock(nn.Module):
    """
    Encoder block:
      - First convolution (with instance normalization and GELU).
      - Followed by edge detection and SC attention (applied to the feature map).
      - Second convolution, then a skip connection is added.
    """
    def __init__(self, in_ch, out_ch):
        super(SCEncoderBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.GELU()
        )
        self.edge = SCEdgeDetectionModule(out_ch)
        self.attn = SCAttention(out_ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.GELU()
        )
        if in_ch != out_ch:
            self.skip_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        feat = self.conv1(x)
        feat = feat + self.edge(feat) + self.attn(feat)
        feat = self.conv2(feat)
        skip = self.skip_conv(x)
        return feat + skip

class SCDecoderBlock(nn.Module):
    """
    Decoder block:
      - Upsamples the input using bilinear interpolation and convolution.
      - Merges with skip connection via concatenation.
      - Refines the merged feature and applies SC attention.
      - Finishes with two convolutions to produce the decoder output.
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.skip_norm = nn.InstanceNorm2d(skip_ch)
        # self.upsample = nn.Sequential(...)   # REMOVE this, or just do Identity

        self.refine = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, in_ch + skip_ch, kernel_size=1),
            nn.InstanceNorm2d(in_ch + skip_ch),
            nn.GELU()
        )
        self.attn = SCAttention(in_ch + skip_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x, skip):
        skip = self.skip_norm(skip)
        # x is already upsampled outside -> shape (B, in_ch, H_up, W_up)

        # Now simply cat
        cat_feat = torch.cat([x, skip], dim=1)
        cat_feat = self.refine(cat_feat)
        cat_feat = self.attn(cat_feat)
        return self.conv(cat_feat)

# ---------------------------------------------------------------------------
# 4. Quaternion Convolution and Cross-Attention Modules
# ---------------------------------------------------------------------------
# For brevity and reliability, we implement a final version of QuaternionConv
# that performs standard quaternion convolution with rotation disabled.
class QuaternionConv(nn.Module):
    """
    Final implementation of Quaternion Convolution.
    Assumes that in_channels and out_channels are divisible by 4.
    Performs standard quaternion convolution (rotation disabled).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(QuaternionConv, self).__init__()
        assert in_channels % 4 == 0, "in_channels must be a multiple of 4"
        assert out_channels % 4 == 0, "out_channels must be a multiple of 4"
        # We partition the input into 4 parts and apply standard Conv2d to each;
        # then combine using quaternion rules.
        self.real = nn.Conv2d(in_channels // 4, out_channels // 4, kernel_size, stride, padding, bias=bias)
        self.i    = nn.Conv2d(in_channels // 4, out_channels // 4, kernel_size, stride, padding, bias=bias)
        self.j    = nn.Conv2d(in_channels // 4, out_channels // 4, kernel_size, stride, padding, bias=bias)
        self.k    = nn.Conv2d(in_channels // 4, out_channels // 4, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        # Split x into 4 channels along the channel axis.
        B, C, H, W = x.shape
        chunk = C // 4
        xr = x[:, :chunk, :, :]
        xi = x[:, chunk:2*chunk, :, :]
        xj = x[:, 2*chunk:3*chunk, :, :]
        xk = x[:, 3*chunk:, :, :]
        # Apply convolution on each sub-channel
        rr = self.real(xr) - self.i(xi) - self.j(xj) - self.k(xk)
        ii = self.i(xr) + self.real(xi) - self.k(xj) + self.j(xk)
        jj = self.j(xr) + self.k(xi) + self.real(xj) - self.i(xk)
        kk = self.k(xr) - self.j(xi) + self.i(xj) + self.real(xk)
        return torch.cat([rr, ii, jj, kk], dim=1)

# Minimal Cross-Attention Transformer (adapted from the SOTA model)
class CrossAttentionTransformer(nn.Module):
    """
    Implements cross-attention followed by a feedforward network.
    Uses PyTorch's built-in MultiheadAttention for efficiency.
    """
    def __init__(self, dim, num_heads=2, bias=True):
        super(CrossAttentionTransformer, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        # Use MultiheadAttention; we flatten spatial dimensions.
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, bias=bias, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, int(dim * 4)),
            nn.GELU(),
            nn.Linear(int(dim * 4), dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, y):
        # x and y: (B, C, H, W); flatten spatial dims: (B, H*W, C)
        B, C, H, W = y.shape
        y_flat = y.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        x_flat = x.view(B, C, H * W).transpose(1, 2)
        # Apply layer norm then MHA: use y as query, x as key/value
        q = self.norm(y_flat)
        attn_out, _ = self.mha(q, x_flat, x_flat)
        y_flat = y_flat + attn_out
        y_flat = y_flat + self.ff(self.norm2(y_flat))
        return y_flat.transpose(1, 2).view(B, C, H, W)

class SelfAttentionTransformer(nn.Module):
    """
    Implements self-attention followed by a feedforward network.
    Uses MultiheadAttention on flattened spatial dimensions.
    """
    def __init__(self, dim, num_heads=2, bias=True):
        super(SelfAttentionTransformer, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, bias=bias, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, int(dim * 4)),
            nn.GELU(),
            nn.Linear(int(dim * 4), dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, C, H, W) -> flatten to (B, H*W, C)
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)
        q = self.norm(x_flat)
        attn_out, _ = self.mha(q, q, q)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.ff(self.norm2(x_flat))
        return x_flat.transpose(1, 2).view(B, C, H, W)

class QCrossAttnBlock(nn.Module):
    """
    Merges cross-attention and self-attention between bottleneck features and a
    downsampled wavelet-color prior, then fuses them via a quaternion convolution.
    This block is crucial for aligning global color information with local features.
    """
    def __init__(self, in_ch=64, heads=2):
        super(QCrossAttnBlock, self).__init__()
        self.cross1 = CrossAttentionTransformer(in_ch, num_heads=heads, bias=True)
        self.cross2 = CrossAttentionTransformer(in_ch, num_heads=heads, bias=True)
        self.self_attn = SelfAttentionTransformer(in_ch, num_heads=heads, bias=True)
        # Ensure in_ch is a multiple of 4 for QuaternionConv.
        if in_ch % 4 != 0:
            raise ValueError("in_ch must be a multiple of 4 for QuaternionConv.")
        self.qconv = nn.Sequential(
            QuaternionConv(in_ch * 4, in_ch * 4, kernel_size=3, padding=1, stride=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch * 4, in_ch, kernel_size=1),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True)
        )

    def forward(self, x, prior):
        # x and prior are (B, in_ch, H, W)
        x1 = self.cross1(prior, x)
        x2 = self.cross2(x, prior)
        x3 = self.self_attn(x)
        z = torch.zeros_like(x)
        cat_feat = torch.cat([z, x1, x2, x3], dim=1)  # (B, 4 * in_ch, H, W)
        return self.qconv(cat_feat)

# ---------------------------------------------------------------------------
# 5. Scale Harmonizer Module
# ---------------------------------------------------------------------------
class ScaleHarmonizer(nn.Module):
    """
    Learns a scale and shift to adaptively merge the original input with
    the network output, ensuring global color consistency.
    """
    def __init__(self, in_nc=6, out_nc=3, base_nf=32):
        super(ScaleHarmonizer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_nc, base_nf, 1)
        self.conv2 = nn.Conv2d(base_nf, base_nf, 1)
        self.conv3 = nn.Conv2d(base_nf, out_nc, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, in_nc, H, W)
        feats = self.pool(x)
        feats = self.act(self.conv1(feats))
        feats = self.act(self.conv2(feats))
        final = self.conv3(feats)
        # Add the learned offset to the first out_nc channels of x.
        out = x[:, :final.shape[1]] + final
        return out

# ---------------------------------------------------------------------------
# 6. Final Unified Underwater Model
# ---------------------------------------------------------------------------
class WaveletModel_V3(nn.Module):
    """
    Fixes the dynamic 'adapter' creation in forward():
      - We define a single 'fused_down_adapter' in __init__ that transforms from 9->(base_ch*4).
      - We remove the second Sigmoid so we only do Sigmoid once at the end.
    """
    def __init__(self, wave='haar', in_ch=3, base_ch=48):
        super().__init__()
        self.prior = SimpleWaveletColorPrior(wave=wave)

        # Initial conv to reduce 9 -> base_ch
        self.init_conv = nn.Sequential(
            nn.Conv2d(9, base_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_ch),
            nn.GELU()
        )

        self.enc1 = SCEncoderBlock(base_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = SCEncoderBlock(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2)
        # Bottleneck
        self.enc3 = SCEncoderBlock(base_ch*2, base_ch*4)

        # We define an adapter for the fused prior to match base_ch*4
        # (We know the prior produces 9 channels initially, then we do init_conv-> base_ch,
        #  but after two pools it is base_ch*2, then another enc => base_ch*4 at bottleneck.)
        self.fused_down_adapter = nn.Sequential(
            nn.Conv2d(9, base_ch*4, kernel_size=1, bias=False),
            nn.InstanceNorm2d(base_ch*4),
            nn.GELU()
        )

        self.cross_attn = QCrossAttnBlock(in_ch=base_ch*4, heads=2)

        # Decoders

        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2 = SCDecoderBlock(in_ch=base_ch*2, skip_ch=base_ch*2, out_ch=base_ch*2)

        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1 = SCDecoderBlock(in_ch=base_ch, skip_ch=base_ch, out_ch=base_ch)

        # Preliminary output
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_ch),
            nn.GELU(),
            nn.Conv2d(base_ch, in_ch, kernel_size=3, padding=1),
            # Only do Sigmoid once at final step, so remove the Sigmoid here:
            # nn.Sigmoid()
        )

        # ScaleHarmonizer merges input & preliminary
        self.harmonizer = ScaleHarmonizer(in_nc=in_ch*2, out_nc=in_ch, base_nf=32)
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1) wavelet+color prior => 9 channels
        fused_in = self.prior(x)  # shape (B, 9, H, W)
        fused_in = check_nan(fused_in, "After wavelet+color prior")

        # 2) init conv => base_ch
        feat0 = self.init_conv(fused_in)
        feat0 = check_nan(feat0, "After initial convolution")

        # 3) Encoders
        e1 = self.enc1(feat0)
        e1 = check_nan(e1, "After encoder block 1")
        p1 = self.pool1(e1)
        p1 = check_nan(p1, "After pooling 1")
        e2 = self.enc2(p1)
        e2 = check_nan(e2, "After encoder block 2")
        p2 = self.pool2(e2)
        p2 = check_nan(p2, "After pooling 2")
        e3 = self.enc3(p2)  # shape (B, base_ch*4, H/4, W/4)
        e3 = check_nan(e3, "After encoder block 3")

        # 4) Also downsample fused_in => shape (B,9,H/4,W/4), then adapt => (B, base_ch*4,H/4,W/4)
        _, _, H3, W3 = e3.shape
        fused_dn = F.interpolate(fused_in, size=(H3, W3), mode='bilinear', align_corners=False)
        fused_dn = self.fused_down_adapter(fused_dn)
        fused_dn = check_nan(fused_dn, "After downsampling and adapting fused input")

        # 5) Cross attention
        bn_out = self.cross_attn(e3, fused_dn)
        bn_out = check_nan(bn_out, "After cross attention")

        # 6) Decoders
        u2 = self.up2(bn_out)  # => (B, base_ch*2, H/2, W/2)
        u2 = check_nan(u2, "After upsampling 2")
        d2 = self.dec2(u2, e2)
        d2 = check_nan(d2, "After decoder block 2")

        u1 = self.up1(d2)      # => (B, base_ch, H, W)
        u1 = check_nan(u1, "After upsampling 1")
        d1 = self.dec1(u1, e1)
        d1 = check_nan(d1, "After decoder block 1")

        # 7) Preliminary output
        out_pre = self.out_conv(d1)    # no Sigmoid here
        out_pre = check_nan(out_pre, "After preliminary output convolution")

        # 8) Scale-harmonize original + out_pre
        combo = torch.cat([x, out_pre], dim=1)  # => (B, in_ch*2, H, W)
        combo = check_nan(combo, "After concatenating original and preliminary output")
        final = self.harmonizer(combo)
        final = check_nan(final, "After scale harmonizer")
        final = self.final_sigmoid(final)       # single Sigmoid at the very end
        final = check_nan(final, "After final sigmoid")

        return final