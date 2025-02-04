#File: models/version2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.random import RandomState
from scipy.stats import chi

try:
    from pytorch_wavelets import DWTForward
except ImportError:
    raise ImportError("Install pytorch_wavelets for wavelet decomposition.")


def check_nan(x, msg):
    """Utility to check for NaNs and print a message. 
       You can comment this out if everything is stable."""
    if torch.isnan(x).any():
        print(f"[NaN Alert] in {msg}")
    return x

def quaternion_init(in_features, out_features, rng, kernel_size=None, criterion='glorot'):
    """
    Same as your code, but carefully used in the final QConv. 
    We assume we won't keep the "rotation=True" for now.
    """
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in = in_features * receptive_field
        fan_out = out_features * receptive_field
    else:
        fan_in = in_features
        fan_out = out_features

    if criterion == 'glorot':
        s = 1. / np.sqrt(2 * (fan_in + fan_out))
    elif criterion == 'he':
        s = 1. / np.sqrt(2 * fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    rng = RandomState(np.random.randint(1, 1234))
    if kernel_size is None:
        kernel_shape = (out_features, in_features)
    else:
        if isinstance(kernel_size, int):
            kernel_shape = (out_features, in_features) + (kernel_size, kernel_size)
        else:
            kernel_shape = (out_features, in_features) + kernel_size

    modulus = chi.rvs(4, loc=0, scale=s, size=kernel_shape)
    number_of_weights = np.prod(kernel_shape)

    v_i = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_j = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_k = np.random.uniform(-1.0, 1.0, number_of_weights)

    for i in range(number_of_weights):
        norm_ijk = np.sqrt(v_i[i]**2 + v_j[i]**2 + v_k[i]**2 + 1e-6)
        v_i[i] /= norm_ijk
        v_j[i] /= norm_ijk
        v_k[i] /= norm_ijk

    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

    weight_r = modulus * np.cos(phase)
    weight_i = modulus * v_i * np.sin(phase)
    weight_j = modulus * v_j * np.sin(phase)
    weight_k = modulus * v_k * np.sin(phase)

    return (weight_r, weight_i, weight_j, weight_k)


def affect_init_conv(r_weight, i_weight, j_weight, k_weight, kernel_size, init_func, rng, init_criterion):
    if (r_weight.size() != i_weight.size() or 
        r_weight.size() != j_weight.size() or 
        r_weight.size() != k_weight.size()):
        raise ValueError("Quaternion weights must have identical shapes.")
    if r_weight.dim() < 3:
        raise ValueError("Expected convolution weights to have >= 3 dims.")

    in_ch = r_weight.size(1)
    out_ch = r_weight.size(0)

    (r_np, i_np, j_np, k_np) = init_func(in_ch, out_ch, rng, kernel_size, init_criterion)

    r_torch = torch.from_numpy(r_np).type_as(r_weight.data)
    i_torch = torch.from_numpy(i_np).type_as(i_weight.data)
    j_torch = torch.from_numpy(j_np).type_as(j_weight.data)
    k_torch = torch.from_numpy(k_np).type_as(k_weight.data)

    r_weight.data.copy_(r_torch)
    i_weight.data.copy_(i_torch)
    j_weight.data.copy_(j_torch)
    k_weight.data.copy_(k_torch)


def quaternion_conv(input, r_w, i_w, j_w, k_w, bias, stride, padding, groups, dilation):
    """
    Standard quaternion 2D conv w/o rotation
    """
    # Construct real-imag part
    cat_kernels_r = torch.cat([r_w, -i_w, -j_w, -k_w], dim=1)
    cat_kernels_i = torch.cat([i_w,  r_w, -k_w,  j_w], dim=1)
    cat_kernels_j = torch.cat([j_w,  k_w,  r_w, -i_w], dim=1)
    cat_kernels_k = torch.cat([k_w, -j_w,  i_w,  r_w], dim=1)
    cat_kernels_4 = torch.cat([cat_kernels_r, cat_kernels_i, cat_kernels_j, cat_kernels_k], dim=0)

    return F.conv2d(input, cat_kernels_4, bias, stride, padding, dilation, groups)


class QuaternionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1, padding=1, groups=1, bias=True,
                 init_criterion='glorot', weight_init=quaternion_init,
                 rotation=False,  # we default to off
                 seed=None):
        super().__init__()
        assert in_channels % 4 == 0, "in_channels must be divisible by 4 for quaternion conv"
        assert out_channels % 4 == 0, "out_channels must be divisible by 4"
        self.in_channels = in_channels // 4
        self.out_channels = out_channels // 4
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.bias_flag = bias
        self.init_criterion = init_criterion
        self.rotation = rotation  # We'll keep it, but default is off

        if seed is None:
            seed = np.random.randint(0, 1234)
        self.rng = RandomState(seed)

        # Weight shapes
        if isinstance(kernel_size, int):
            ks = (kernel_size, kernel_size)
        else:
            ks = kernel_size
        w_shape = (out_channels // 4, in_channels // 4) + ks

        # Quaternion parts
        self.r_weight = nn.Parameter(torch.Tensor(*w_shape))
        self.i_weight = nn.Parameter(torch.Tensor(*w_shape))
        self.j_weight = nn.Parameter(torch.Tensor(*w_shape))
        self.k_weight = nn.Parameter(torch.Tensor(*w_shape))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters(weight_init)

    def reset_parameters(self, winit):
        affect_init_conv(
            self.r_weight, self.i_weight, self.j_weight, self.k_weight,
            self.kernel_size, winit, self.rng, self.init_criterion
        )
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # We do NOT apply rotation-based conv by default now.
        # Just the standard quaternion convolution:
        out = quaternion_conv(
            x, 
            self.r_weight, self.i_weight, self.j_weight, self.k_weight,
            self.bias, 
            stride=self.stride, padding=self.padding, 
            groups=self.groups, dilation=self.dilation
        )
        return out


# 2) Wavelet + FFT Decomposition
# -----------------------------------
# We remove the harsh "clamp_subband" usage.

try:
    from pytorch_wavelets import DWTForward
except ImportError:
    raise ImportError("Please install 'pytorch_wavelets' to use this Wavelet Decomposition.")


class WaveletFFTDecomposition(nn.Module):
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.dwt = DWTForward(J=1, wave=wavelet)

    def forward(self, x):
        x = x.float()  # Ensure float32
        Yl, Yh = self.dwt(x)   # Yl: (B,C,H/2,W/2), Yh[0]: (B,C,3,H/2,W/2)
        LL = Yl
        LH = Yh[0][:, :, 0]
        HL = Yh[0][:, :, 1]
        HH = Yh[0][:, :, 2]

        # FFT
        fft_complex = torch.fft.fft2(x, dim=(-2, -1))
        mag = torch.abs(fft_complex)
        phase = torch.angle(fft_complex)

        return LL, LH, HL, HH, mag, phase


# 3) Sub-Band Attention (Simplified: no zeroing out)
# ---------------------------------------------------
class SubBandAttention(nn.Module):
    def __init__(self, num_bands=3):
        super().__init__()
        # Just learnable gating for LH, HL, HH
        self.attn_params = nn.Parameter(torch.ones(num_bands))  # init = 1
    def forward(self, lh, hl, hh):
        # No forced zeroing for tiny values.
        raw = self.attn_params
        # Use a small stable range
        attn = 0.1 + 0.9 * torch.sigmoid(raw)  # so it's in [0.1, 1.0]
        lh_out = lh * attn[0]
        hl_out = hl * attn[1]
        hh_out = hh * attn[2]
        return lh_out, hl_out, hh_out


# A simple path that processes sub-bands at half resolution
class LowResWaveletPath(nn.Module):
    def __init__(self, in_ch=3, mid_ch=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_ch, in_ch, 3, padding=1)
        # We only do instance norm once
        self.norm = nn.InstanceNorm2d(in_ch, affine=True)

    def forward(self, x):
        out = self.act1(self.conv1(x))
        out = self.conv2(out)
        out = self.norm(out)
        return out


# 4) A simpler FFT branch
# ---------------------------------------------------
class FFTBranch(nn.Module):
    def __init__(self, in_ch=3, mid_ch=16, out_ch=6):
        super().__init__()
        self.mag_net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, in_ch, 3, padding=1)
        )
        self.phase_net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, in_ch, 3, padding=1)
        )
        self.fusion = nn.Conv2d(in_ch * 2, out_ch, kernel_size=1)
    def forward(self, mag, phase):
        mag_feat = self.mag_net(mag)
        phase_feat = self.phase_net(phase)
        comb = torch.cat([mag_feat, phase_feat], dim=1)
        out = self.fusion(comb)
        return out


# 5) A simpler color prior (like your MultiStageColorPrior but fewer prints)
# --------------------------------------------------------------------------
class MultiStageColorPrior(nn.Module):
    def __init__(self, in_ch=3, hidden_dim=16, unet_ch=16):
        super().__init__()
        self.in_ch = in_ch
        # (A) global shift
        self.global_linear = nn.Sequential(
            nn.Linear(in_ch, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_ch)
        )
        # (B) local offset (U-Net)
        self.color_unet = ColorUNet(in_ch, base_ch=unet_ch)

    def forward(self, x):
        B, C, H, W = x.shape
        # Global shift
        avg_color = x.view(B, C, -1).mean(dim=2)  # (B,3)
        shift = self.global_linear(avg_color).view(B, C, 1, 1)
        global_corrected = x + shift
        # Local offset
        color_offset = self.color_unet(global_corrected)
        color_prior = global_corrected + color_offset
        return color_prior

class ColorUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=16):
        super().__init__()
        # standard small U-Net
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(True),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1),
            nn.ReLU(True),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*4, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(base_ch*4, base_ch*4, 3, padding=1),
            nn.ReLU(True),
        )
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*2, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1),
            nn.ReLU(True),
        )
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(True),
        )
        self.out_conv = nn.Conv2d(base_ch, in_ch, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b  = self.bottleneck(p2)
        u2 = self.up2(b)
        cat2 = torch.cat([u2,e2], dim=1)
        d2 = self.dec2(cat2)
        u1 = self.up1(d2)
        cat1 = torch.cat([u1,e1], dim=1)
        d1 = self.dec1(cat1)
        offset = self.out_conv(d1)
        return offset


# 6) Quaternion trunk (simplified)
# ----------------------------------
class SimpleQuaternionFeatureExtractor(nn.Module):
    def __init__(self, in_ch=3, out_ch=32):
        super().__init__()
        # We keep a single "adapter" for quaternion:
        # in_ch=3 => we replicate or transform to 4 channels
        # but let's do a simpler approach: we require the input to be multiple of 4
        # So let's do: if in_ch=3, we first do a 1x1 conv to 4 channels
        # Then one or two quaternion blocks => final out
        # out_ch must be multiple of 4.
        assert out_ch % 4 == 0, "out_ch must be multiple of 4"
        self.adapter = nn.Conv2d(in_ch, 4, kernel_size=1, bias=False)
        # QConv1
        self.q1 = QuaternionConv(4, out_ch, kernel_size=3, padding=1, stride=1, bias=True, rotation=False)
        self.norm1 = nn.InstanceNorm2d(out_ch, affine=True)
        self.act1 = nn.ReLU(inplace=True)
        # QConv2
        self.q2 = QuaternionConv(out_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=True, rotation=False)
        self.norm2 = nn.InstanceNorm2d(out_ch, affine=True)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x_adapt = self.adapter(x)    # => (B,4,H,W)
        out = self.q1(x_adapt)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.q2(out)
        out = self.norm2(out)
        out = self.act2(out)
        return out


# 7) Cross-Attention + FeedForward
# ----------------------------------
class FeedForward(nn.Module):
    def __init__(self, dim_in, expansion_factor=2.0):
        super().__init__()
        hidden = int(dim_in * expansion_factor)
        self.conv1 = nn.Conv2d(dim_in, hidden, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.conv2 = nn.Conv2d(hidden, dim_in, kernel_size=1)
    def forward(self, x):
        inp = x
        x = self.conv1(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + inp

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=2):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads,1,1))
        self.kv = nn.Conv2d(dim, dim*2, 1)
        self.q  = nn.Conv2d(dim, dim, 1)
        self.proj_out = nn.Conv2d(dim, dim, 1)
    def forward(self, x_q, x_k):
        B,C,H,W = x_q.shape
        kv = self.kv(x_k)
        k, v = kv.chunk(2, dim=1)
        q = self.q(x_q)

        # reshape
        def reshape_heads(tensor):
            return tensor.view(B, self.heads, C//self.heads, H*W)
        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # normalize
        q = F.normalize(q, dim=2)
        k = F.normalize(k, dim=2)

        # Lower attn scale
        attn_scale = 0.05 * torch.sigmoid(self.temperature)  # in [0,0.05]
        attn = torch.einsum('bhcd,bhkd->bhck', q, k) * attn_scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhck,bhkd->bhcd', attn, v)
        out = out.reshape(B, C, H, W)
        out = self.proj_out(out)
        return out

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=2, ffn_factor=2.0):
        super().__init__()
        self.norm_x = nn.InstanceNorm2d(dim, affine=True)
        self.norm_y = nn.InstanceNorm2d(dim, affine=True)
        self.cross_attn = CrossAttention(dim, heads)
        self.ffn = FeedForward(dim, expansion_factor=ffn_factor)
    def forward(self, x, y):
        # typical usage: y = cross_attn(x, y)
        # We'll do: y <- y + attn(x,y)
        y_in = y
        attn_out = self.cross_attn(self.norm_x(x), self.norm_y(y))
        y = y_in + attn_out
        y = self.ffn(y)
        return y


# 8) Simple DeeperDetailRestorer
# --------------------------------
class QBlock(nn.Module):
    """
    A small quaternion block: QConv -> ReLU -> QConv -> skip
    Using instance norm only once or none.
    """
    def __init__(self, ch):
        super().__init__()
        self.q1 = QuaternionConv(ch, ch, 3, padding=1, bias=True, rotation=False)
        self.act1 = nn.ReLU(inplace=True)
        self.q2 = QuaternionConv(ch, ch, 3, padding=1, bias=True, rotation=False)
        self.norm = nn.InstanceNorm2d(ch, affine=True)
    def forward(self, x):
        res = x
        out = self.q1(x)
        out = self.act1(out)
        out = self.q2(out)
        out = self.norm(out)
        return out + res

class DeeperDetailRestorer(nn.Module):
    def __init__(self, ch=32, num_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList([QBlock(ch) for _ in range(num_blocks)])
        self.final_act = nn.ReLU(inplace=True)
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.final_act(x)
        return x


# 9) Spatial Pyramid Pooling
# ---------------------------
class SPP(nn.Module):
    """
    Simple multi-scale avg-pooling + upsample, then fuse.
    """
    def __init__(self, in_ch, out_ch, num_levels=3):
        super().__init__()
        self.num_levels = num_levels
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, in_ch, 1, bias=False) for _ in range(self.num_levels)
        ])
        self.fusion = nn.Conv2d(in_ch*(num_levels+1), out_ch, 3, padding=1, bias=False)
    def forward(self, x):
        B,C,H,W = x.shape
        out_list = []
        for i in range(self.num_levels):
            k = 2**(i+1)
            # pool
            pooled = F.avg_pool2d(x, kernel_size=k, stride=k)
            # 1x1 conv
            feat = self.convs[i](pooled)
            # upsample
            up = F.interpolate(feat, size=(H,W), mode='bilinear', align_corners=False)
            out_list.append(up)
        out_list.append(x)
        cat_out = torch.cat(out_list, dim=1)
        fused = self.fusion(cat_out)
        return fused


# 10) Final Output ScaleShift
# ----------------------------
class FinalScaleShift(nn.Module):
    def __init__(self, num_channels=3):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
    def forward(self, x):
        return x * self.scale + self.shift


# 11) The Final Wavelet Model
# ----------------------------
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class WaveletModel_V2(nn.Module):
    def __init__(self,
                 base_ch=32,
                 wavelet='haar',
                 use_subband_attn=True,
                 use_fft_branch=True,
                 deeper_detail=True):
        super().__init__()
        assert base_ch % 4 == 0, "base_ch must be multiple of 4 for quaternion convs."
        self.use_subband_attn = use_subband_attn
        self.use_fft_branch   = use_fft_branch

        # 1) color prior
        self.color_prior = MultiStageColorPrior(in_ch=3, hidden_dim=16, unet_ch=16)

        # 2) wavelet + FFT decomposition
        self.spectral_decomp = WaveletFFTDecomposition(wavelet=wavelet)

        if use_subband_attn:
            self.subband_attn = SubBandAttention(num_bands=3)
            self.lowres_path = LowResWaveletPath(in_ch=3, mid_ch=16)

        if use_fft_branch:
            self.fft_branch = FFTBranch(in_ch=3, mid_ch=16, out_ch=6)

        # 3) quaternion trunk
        self.qfeat = SimpleQuaternionFeatureExtractor(in_ch=3, out_ch=base_ch)

        # 4) cross-attn blocks
        self.cross_block1 = CrossAttentionBlock(dim=base_ch, heads=2, ffn_factor=2.0)
        self.cross_block2 = CrossAttentionBlock(dim=base_ch, heads=2, ffn_factor=2.0)

        # 5) detail restoration
        if deeper_detail:
            self.detail_restorer = DeeperDetailRestorer(ch=base_ch, num_blocks=3)
        else:
            # If user wants simpler
            self.detail_restorer = DeeperDetailRestorer(ch=base_ch, num_blocks=1)

        # 6) SPP
        self.spp = SPP(in_ch=base_ch, out_ch=base_ch, num_levels=3)

        # 7) final
        self.final_out = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)
        self.final_harmonizer = FinalScaleShift(num_channels=3)
        self.sigmoid = nn.Sigmoid()

        # unify dims
        # color:  (B,3,H,W) -> (B, base_ch,H,W) with 1x1
        self.color_reduce = nn.Conv2d(3, base_ch, kernel_size=1, bias=False)
        # wavelet + fft might produce up to (3+3+3+3+6)=18 channels
        self.spec_reduce = nn.Sequential(
            nn.Conv2d(18, base_ch*2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_ch*2, base_ch, 3, padding=1)
        )

        self.apply(init_weights)

    def forward(self, x):
        B, C, H, W = x.shape
        # 1) color prior
        color_feat = check_nan(self.color_prior(x), "color_prior")

        # 2) wavelet + FFT
        LL, LH, HL, HH, mag, phase = self.spectral_decomp(x)
        LH, HL, HH = check_nan(LH, "LH"), check_nan(HL, "HL"), check_nan(HH, "HH")
        mag, phase = check_nan(mag, "mag"), check_nan(phase, "phase")

        if self.use_subband_attn:
            LH, HL, HH = self.subband_attn(LH, HL, HH)
            LH_proc = self.lowres_path(check_nan(LH, "LH_after_attn"))
            HL_proc = self.lowres_path(check_nan(HL, "HL_after_attn"))
            HH_proc = self.lowres_path(check_nan(HH, "HH_after_attn"))
            LH_up = F.interpolate(LH_proc, (H, W), mode='bilinear', align_corners=False)
            HL_up = F.interpolate(HL_proc, (H, W), mode='bilinear', align_corners=False)
            HH_up = F.interpolate(HH_proc, (H, W), mode='bilinear', align_corners=False)
            LL_up = F.interpolate(check_nan(LL, "LL"), (H, W), mode='bilinear', align_corners=False)
        else:
            LH_up = F.interpolate(LH, (H,W), mode='bilinear')
            HL_up = F.interpolate(HL, (H,W), mode='bilinear')
            HH_up = F.interpolate(HH, (H,W), mode='bilinear')
            LL_up = F.interpolate(LL, (H,W), mode='bilinear')

        if self.use_fft_branch:
            fft_out = check_nan(self.fft_branch(mag, phase), "fft_branch_out")
            spectral_feat = torch.cat([LL_up, LH_up, HL_up, HH_up, fft_out], dim=1)
        else:
            spectral_feat = torch.cat([LL_up, LH_up, HL_up, HH_up, mag, phase], dim=1)
        spectral_feat = check_nan(spectral_feat, "spectral_feat")

        # 3) trunk feat
        trunk_feat = check_nan(self.qfeat(x), "trunk_feat")

        # 4) cross-attn merges trunk & color
        color_aligned = check_nan(self.color_reduce(color_feat), "color_reduce")
        y1 = check_nan(self.cross_block1(color_aligned, trunk_feat), "cross_block1")

        # cross-attn merges trunk & wavelet
        spec_aligned = check_nan(self.spec_reduce(spectral_feat), "spec_reduce")
        y2 = check_nan(self.cross_block2(y1, spec_aligned), "cross_block2")

        # 5) detail restoration
        detail_out = check_nan(self.detail_restorer(y2), "detail_restorer")

        # 6) SPP
        spp_out = check_nan(self.spp(detail_out), "spp_out")

        # 7) final
        out = check_nan(self.final_out(spp_out), "final_conv")
        out = check_nan(self.final_harmonizer(out), "final_harmonizer")
        out = torch.sigmoid(out)
        out = check_nan(out, "sigmoid_out")

        return out