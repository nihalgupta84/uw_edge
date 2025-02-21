# File: models/edge_model_v1.py


"""
edge_model_v1.py

This file implements a configurable underwater enhancement or haze removal model.
It is built around the `SCBackbone` architecture, which includes:

1) Multiple Encoder Blocks with optional:
   - Edge Detection (SCEdgeDetectionModule)
   - Self-Attention (SCAttention)

2) A Bottleneck combining the highest-level features.

3) Multiple Decoder Blocks that:
   - Optionally fuse high-level + skip features using CGAFusion
     or a simpler concatenation-based fusion
   - Optionally include a Self-Attention step

Finally, a 1×1 or 3×3 'final' convolution produces the restored image.

------------------------------------------
I. SCEdgeDetectionModule
   - Takes an input of shape [B, C, H, W]
   - If use_ck, use_hk, use_vk are True, it applies 3 depthwise kernels
     for center difference + sobel horizontal + sobel vertical.
     Otherwise, it returns a zero map for each disabled branch.
   - Concatenates the 3 outputs => [B, 3C, H, W], then applies a 1×1 conv
     down to [B, C, H, W].
   - Goal: capture edge/gradient details in a channel-wise manner.

------------------------------------------
II. SCAttention
   - Channel Attention
       * A global average pool => [B, C, 1, 1], followed by MLP => scale factor
         that is broadcast across spatial dims => [B, C, H, W].
   - Spatial Attention
       * Takes min/max across channels => [B,1,H,W], merges => [B,2,H,W],
         then a 7×7 conv => [B,1,H,W] => merges into the original feature map.
   - Edge Enhancement
       * A depthwise 3×3 conv => [B, C, H, W].
   - Fusion
       * Concats channel_enhanced + spatial_enhanced => [B,2C,H,W]
         => 1×1 conv => [B,C,H,W], then add edge_enhanced => final [B, C, H, W].

------------------------------------------
III. SCEncoderBlock
   Input : [B, in_channels,  H, W]
   Output: [B, out_channels, H, W]
   Steps:
   1) First conv => [B, out_channels, H, W]
   2) Optional Edge detection => [B, out_channels, H, W]
      Optional Attention => [B, out_channels, H, W]
      Sum them with the 'out' => new_out
   3) Dropout + second conv => [B, out_channels, H, W]
   4) Residual skip if in_channels != out_channels => ensures shape matches
      final => [B, out_channels, H, W]

------------------------------------------
IV. SCDecoderBlock
   Input : x => [B, in_channels,   H_dec, W_dec]
           skip => [B, skip_ch, H_dec, W_dec]
   Output: [B, out_channels, 2×H_dec, 2×W_dec] (after upsample)
   Actually, the upsample step is inside the block, so the shapes might shift:
     - Step1: Upsample x => [B, in_channels, 2×H_dec, 2×W_dec]
     - Step2: If use_cgafusion, align skip => [B, in_channels, 2×H_dec, 2×W_dec]
              then CGAFusion => returns [B, in_channels, 2×H_dec, 2×W_dec]
       Else, cat => [B, in_channels + skip_ch, 2×H_dec, 2×W_dec]
     - Step3: 'refine' => [B, mid_channels, 2×H_dec, 2×W_dec]
     - Step4: optional SCAttention => [B, mid_channels, 2×H_dec, 2×W_dec]
     - Step5: final conv => [B, out_channels, 2×H_dec, 2×W_dec]

------------------------------------------
V. SCBackbone
   - Overall U-Net style architecture (encoder → bottleneck → decoder).
   - Encoders:
     * encoder1 => [B, base_ch, H, W]
       downsample => [B, base_ch, H/2, W/2]
     * encoder2 => [B, 2*base_ch, H/2, W/2]
       downsample => [B, 2*base_ch, H/4, W/4]
     * encoder3 => [B, 4*base_ch, H/4, W/4]
       downsample => [B, 4*base_ch, H/8, W/8]
   - Bottleneck => [B, 8*base_ch, H/8, W/8]
   - Decoders merge skip connections:
     * decoder3 merges with s3 => output [B, 4*base_ch, H/4, W/4]
     * decoder2 merges with s2 => output [B, 2*base_ch, H/2, W/2]
     * decoder1 merges with s1 => output [B, base_ch,   H,   W]
   - final => [B, in_ch, H, W]

------------------------------------------
VI. EdgeModel_V1
   - A simple wrapper that instantiates `SCBackbone(...)` with
     user-defined config flags for edge detection, attention, CGAFusion, etc.
   - forward(x) returns only the final restored image => shape [B, in_ch, H, W]
   - _get_featurehr(x) returns the last decoder feature => shape [B, base_ch, H, W]

------------------------------------------
Shape Summary:
 - Input: [B, 3, H, W]
 - Through each encoder: resolution halves, channel dimension typically doubles
 - Bottleneck: [B, 8*base_ch, H/8, W/8]
 - Decoders: resolution doubles each step, channels go from 8*base_ch → 4*base_ch → 2*base_ch → base_ch
 - Final layer: [B, in_ch, H, W]

"""


import torch
import warnings
from thop import profile, clever_format
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from utils.fusion import CGAFusion
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


###############################################################################
# 1. SCEdgeDetectionModule
###############################################################################
class SCEdgeDetectionModule(nn.Module):
    """
    Optional Edge Detection using custom kernels (Center Difference + Sobel variants).
    """
    def __init__(self, channels, use_ck=True, use_hk=True, use_vk=True):
        super(SCEdgeDetectionModule, self).__init__()
        self.use_ck = use_ck
        self.use_hk = use_hk
        self.use_vk = use_vk

        # Depthwise conv with weight normalization
        self.cdc = nn.utils.weight_norm(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        )
        self.hdc = nn.utils.weight_norm(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        )
        self.vdc = nn.utils.weight_norm(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        )

        # Fuse the 3 edge maps
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.InstanceNorm2d(channels),
            nn.GELU()
        )

        # Initialize custom kernels
        self._init_edge_kernels()

    def _init_edge_kernels(self):
        # Center difference kernel
        cdc_kernel = torch.zeros(1, 1, 3, 3)
        cdc_kernel[0, 0, 1, 1] = 1
        cdc_kernel[0, 0, :, :] -= 1/8
        epsilon = 1e-5
        cdc_kernel = cdc_kernel / (cdc_kernel.abs().sum() + epsilon)

        # Sobel horizontal
        hdc_kernel = torch.tensor([
            [-1,  0, 1],
            [-2,  0, 2],
            [-1,  0, 1]
        ]).float().view(1, 1, 3, 3) / 8.0

        # Sobel vertical
        vdc_kernel = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ]).float().view(1, 1, 3, 3) / 8.0

        self.register_buffer('cdc_kernel', cdc_kernel)
        self.register_buffer('hdc_kernel', hdc_kernel)
        self.register_buffer('vdc_kernel', vdc_kernel)

        # Assign weights
        self.cdc.weight.data = cdc_kernel.repeat(self.cdc.weight.shape[0], 1, 1, 1)
        self.hdc.weight.data = hdc_kernel.repeat(self.hdc.weight.shape[0], 1, 1, 1)
        self.vdc.weight.data = vdc_kernel.repeat(self.vdc.weight.shape[0], 1, 1, 1)

    def forward(self, x):
        # If flags are off, we output zeros for that branch
        cdc_out = self.cdc(x) if self.use_ck else torch.zeros_like(x)
        hdc_out = self.hdc(x) if self.use_hk else torch.zeros_like(x)
        vdc_out = self.vdc(x) if self.use_vk else torch.zeros_like(x)

        edge_feats = torch.cat([cdc_out, hdc_out, vdc_out], dim=1)  # [B, 3C, H, W]
        return self.fusion(edge_feats)                              # [B, C, H, W]


###############################################################################
# 2. SCAttention
###############################################################################
class SCAttention(nn.Module):
    """
    Self-Attention combining channel-wise and spatial attention, plus edge enhancement.
    """
    def __init__(self, channels, reduction=8):
        super(SCAttention, self).__init__()

        # Channel attention
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.LayerNorm([channels, 1, 1]),
            nn.Conv2d(channels, channels//reduction, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels//reduction, channels, kernel_size=1)
        )
        self.layer_norm = nn.LayerNorm([channels, 1, 1])

        # Spatial attention
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )

        # Edge-like enhancement
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.InstanceNorm2d(channels),
            nn.GELU()
        )

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.InstanceNorm2d(channels),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Channel attention => [B, C, 1, 1]
        c_att = self.channel_gate(x)
        c_att = self.layer_norm(c_att)
        channel_enhanced = x * c_att.sigmoid()

        # Spatial attention => [B, 1, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        s_in = torch.cat([avg_out, max_out], dim=1)
        s_att = self.spatial_gate(s_in)
        spatial_enhanced = x * s_att

        # Edge
        edge_enhanced = self.edge_conv(x)

        # Combine channel & spatial => [B, 2C, H, W]
        combined = torch.cat([channel_enhanced, spatial_enhanced], dim=1)
        fused = self.fusion(combined) + edge_enhanced  # => [B, C, H, W]
        return fused


###############################################################################
# 3. SCDecoderBlock
###############################################################################
class SCDecoderBlock(nn.Module):
    """
    Decoder block that can optionally use CGAFusion or old skip+cat fusion,
    and optionally apply attention at the decoder stage.
    """
    def __init__(
        self,
        in_channels,         # Channels of the upsampled feature
        skip_channels,       # Channels from the skip connection
        out_channels,        # Output channels after the final conv
        use_cgafusion=True,  # Whether to apply CGAFusion or simple cat
        use_attention_dec=True
    ):
        super(SCDecoderBlock, self).__init__()

        self.use_cgafusion = use_cgafusion
        self.use_attention_dec = use_attention_dec

        # Normalize skip
        self.skip_norm = nn.InstanceNorm2d(skip_channels)

        # Upsample pipeline for 'x'
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.GELU()
        )

        if self.use_cgafusion:
            # CGAFusion requires both features to have the same channels => in_channels
            if skip_channels != in_channels:
                self.skip_align = nn.Conv2d(skip_channels, in_channels, kernel_size=1)
            else:
                self.skip_align = nn.Identity()

            # CGAFusion (channel = in_channels)
            self.fusion = CGAFusion(dim=in_channels, reduction=8)
            mid_channels = in_channels
        else:
            # Old skip+cat approach => concatenates [in_channels + skip_channels]
            self.skip_align = nn.Identity()  # Not needed
            self.fusion = None
            mid_channels = in_channels + skip_channels

        # Refine
        self.refine = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.InstanceNorm2d(mid_channels),
            nn.GELU()
        )

        # Optional attention in the decoder
        if self.use_attention_dec:
            self.attention = SCAttention(mid_channels)
        else:
            self.attention = nn.Identity()

        # Final conv => [B, out_channels, H, W]
        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x, skip):
        # 1) Normalize skip
        skip = self.skip_norm(skip)

        # 2) Upsample x => [B, in_channels, H, W]
        x = self.upsample(x)

        if self.use_cgafusion:
            # Align skip => [B, in_channels, H, W]
            skip = self.skip_align(skip)
            # CGAFusion => [B, in_channels, H, W]
            fused = self.fusion(x, skip)
        else:
            # Old approach => simply cat => [B, in_channels + skip_channels, H, W]
            fused = torch.cat([x, skip], dim=1)

        # Refine => [B, mid_channels, H, W]
        x = self.refine(fused)
        # Decoder attention
        x = self.attention(x)
        # Final conv => [B, out_channels, H, W]
        x = self.conv(x)
        return x


###############################################################################
# 4. SCEncoderBlock
###############################################################################
class SCEncoderBlock(nn.Module):
    """
    Encoder block with optional EdgeDetection + optional SCAttention + conv layers.
    """
    def __init__(self, in_channels, out_channels,
                 use_edge_module=True,
                 use_attention_module=True,
                 use_ck=True, use_hk=True, use_vk=True):
        super(SCEncoderBlock, self).__init__()
        self.use_edge_module = use_edge_module
        self.use_attention_module = use_attention_module

        # First conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU()
        )

        # Edge detection
        if self.use_edge_module:
            self.edge_detect = SCEdgeDetectionModule(
                out_channels, use_ck=use_ck, use_hk=use_hk, use_vk=use_vk
            )
        else:
            self.edge_detect = nn.Identity()

        # Attention
        if self.use_attention_module:
            self.attention = SCAttention(out_channels)
        else:
            self.attention = nn.Identity()

        # Second conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU()
        )

        # If in/out differ, we add a 1x1 to unify for the residual
        if in_channels != out_channels:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.InstanceNorm2d(out_channels)
            )
        else:
            self.skip_conv = nn.Identity()

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # First conv
        out = self.conv1(x)

        # Edge + attention
        e_feats = self.edge_detect(out) if self.use_edge_module else torch.zeros_like(out)
        a_feats = self.attention(out)   if self.use_attention_module else torch.zeros_like(out)
        out = out + e_feats + a_feats

        out = self.dropout(out)
        out = self.conv2(out)

        # Residual skip
        skip = self.skip_conv(x)
        out = out + skip
        return out

###############################################################################
# 5. Weight Initialization
###############################################################################
def init_model_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

###############################################################################
# 6. SCBackbone
###############################################################################
class SCBackbone(nn.Module):
    """
    sjsbjsb
    """
    def __init__(self,
                 in_ch=3,
                 base_ch=64,
                 use_edge_module=True,
                 use_attention_module=True,
                 use_ck=True,
                 use_hk=True,
                 use_vk=True,
                 use_cgafusion=True,
                 use_attention_dec=True,
                 init_weights=True):
        super(SCBackbone, self).__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=7, padding=3),
            nn.InstanceNorm2d(base_ch),
            nn.GELU(),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_ch),
            nn.GELU()
        )

        # ----------------- Encoders -----------------
        self.encoder1 = SCEncoderBlock(base_ch, base_ch,
                                       use_edge_module=use_edge_module,
                                       use_attention_module=use_attention_module,
                                       use_ck=use_ck, use_hk=use_hk, use_vk=use_vk)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), nn.Dropout(0.1))

        self.encoder2 = SCEncoderBlock(base_ch, base_ch * 2,
                                       use_edge_module=use_edge_module,
                                       use_attention_module=use_attention_module,
                                       use_ck=use_ck, use_hk=use_hk, use_vk=use_vk)
        self.down2 = nn.Sequential(nn.MaxPool2d(2), nn.Dropout(0.1))

        self.encoder3 = SCEncoderBlock(base_ch * 2, base_ch * 4,
                                       use_edge_module=use_edge_module,
                                       use_attention_module=use_attention_module,
                                       use_ck=use_ck, use_hk=use_hk, use_vk=use_vk)
        self.down3 = nn.Sequential(nn.MaxPool2d(2), nn.Dropout(0.1))

        # --------------- Bottleneck ---------------
        self.bottleneck = nn.Sequential(
            SCEncoderBlock(base_ch * 4, base_ch * 8,
                           use_edge_module=use_edge_module,
                           use_attention_module=use_attention_module,
                           use_ck=use_ck, use_hk=use_hk, use_vk=use_vk),
            SCAttention(base_ch * 8),
            nn.Dropout(0.2)
        )

        # ----------------- Decoders -----------------
        # Use SCDecoderBlock with the new toggles
        self.decoder3 = SCDecoderBlock(base_ch * 8, base_ch * 4, base_ch * 4,
                                       use_cgafusion=use_cgafusion,
                                       use_attention_dec=use_attention_dec)
        self.decoder2 = SCDecoderBlock(base_ch * 4, base_ch * 2, base_ch * 2,
                                       use_cgafusion=use_cgafusion,
                                       use_attention_dec=use_attention_dec)
        self.decoder1 = SCDecoderBlock(base_ch * 2, base_ch, base_ch,
                                       use_cgafusion=use_cgafusion,
                                       use_attention_dec=use_attention_dec)

        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_ch),
            nn.GELU(),
            nn.Conv2d(base_ch, in_ch, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        if init_weights:
            self.apply(init_model_weights)

    def forward(self, x):
        # 1) Initial convolution
        x0 = self.init_conv(x)

        # 2) Encoder stages
        s1 = self.encoder1(x0)
        x1 = self.down1(s1)

        s2 = self.encoder2(x1)
        x2 = self.down2(s2)

        s3 = self.encoder3(x2)
        x3 = self.down3(s3)

        # 3) Bottleneck
        b = self.bottleneck(x3)

        # 4) Decoder stages
        d3 = self.decoder3(b, s3)
        d2 = self.decoder2(d3, s2)
        d1 = self.decoder1(d2, s1)

        # 5) Final output
        out = self.final(d1)
        return d1, out


###############################################################################
# 7. EdgeModel_V1
###############################################################################
class EdgeModel_V1(nn.Module):
    """
    Example wrapper model that uses SCBackbone for underwater/haze correction.
    """
    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 use_edge_module=True,
                 use_attention_module=True,
                 use_ck=True, use_hk=True, use_vk=True,
                 use_cgafusion=True, use_attention_dec=True,
                 init_weights=True):
        super(EdgeModel_V1, self).__init__()

        self.backbone = SCBackbone(
            in_ch=in_channels,
            base_ch=base_channels,
            use_edge_module=use_edge_module,
            use_attention_module=use_attention_module,
            use_ck=use_ck,
            use_hk=use_hk,
            use_vk=use_vk,
            use_cgafusion=use_cgafusion,
            use_attention_dec=use_attention_dec,
            init_weights=init_weights
        )

    def forward(self, x):
        """
        Returns just the final 'restored' image (hazeRemoval), or any name you prefer.
        """
        _, out = self.backbone(x)
        return out

    def _get_featurehr(self, x):
        """
        If you need the decoder features for something else, call this method.
        """
        featrueHR, _ = self.backbone(x)
        return featrueHR

    
# def test_model():
#     model = EdgeModel().to('cuda')
#     inp = torch.randn(1, 3, 256, 256).to('cuda')
#     print("Model Summary:")
#     summary(model, input_size=(1, 3, 256, 256))

#     out = model(inp)
#     print(out.shape)

# if __name__ == '__main__':
#     from thop import profile
#     from thop import clever_format
#     from torchinfo import summary
#     inp = torch.randn(1, 3, 256, 256)
#     model = EdgeModel()
#     macs, params = profile(model, inputs=(inp,))
#     macs, params = clever_format([macs, params], "%.3f")
#     print(macs, params)
#     test_model()


def test_scbackbone():
    model = SCBackbone().to('cuda')
    inp = torch.randn(1, 3, 256, 256).to('cuda')
    summary(model, input_size=(1, 3, 256, 256))
    # Forward pass with intermediate shape prints
    print(f"/n scbackbone")
    x0 = model.init_conv(inp)
    print("After init_conv:", x0.shape)

    s1 = model.encoder1(x0)
    print("After encoder1:", s1.shape)
    x1 = model.down1(s1)
    print("After down1:", x1.shape)

    s2 = model.encoder2(x1)
    print("After encoder2:", s2.shape)
    x2 = model.down2(s2)
    print("After down2:", x2.shape)

    s3 = model.encoder3(x2)
    print("After encoder3:", s3.shape)
    x3 = model.down3(s3)
    print("After down3:", x3.shape)

    b = model.bottleneck(x3)
    print("After bottleneck:", b.shape)

    d3 = model.decoder3(b, s3)
    print("After decoder3:", d3.shape)
    d2 = model.decoder2(d3, s2)
    print("After decoder2:", d2.shape)
    d1 = model.decoder1(d2, s1)
    print("After decoder1:", d1.shape)

    out = model.final(d1)
    print("After final:", out.shape)

    macs, params = profile(model, inputs=(inp,))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"MACs: {macs}, Params: {params}")

def test_scencoderblock():
    model = SCEncoderBlock(3, 64).to('cuda')
    inp = torch.randn(1, 3, 256, 256).to('cuda')
    print(f"/n scencoderblock")
    out = model.conv1(inp)
    print("After conv1:", out.shape)

    e_feats = model.edge_detect(out)
    print("After edge_detect:", e_feats.shape)

    a_feats = model.attention(out)
    print("After attention:", a_feats.shape)

    out = out + e_feats + a_feats
    out = model.dropout(out)
    print("After dropout:", out.shape)

    out = model.conv2(out)
    print("After conv2:", out.shape)

    skip = model.skip_conv(inp)
    out = out + skip
    print("After skip connection:", out.shape)

    macs, params = profile(model, inputs=(inp,))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"MACs: {macs}, Params: {params}")

def test_scdecoderblock():
    model = SCDecoderBlock(128, 64, 64).to('cuda')
    inp = torch.randn(1, 128, 64, 64).to('cuda')
    skip = torch.randn(1, 64, 128, 128).to('cuda')
    print(f"/n scdecoderblock")
    skip = model.skip_norm(skip)
    print("After skip_norm:", skip.shape)

    x = model.upsample(inp)
    print("After upsample:", x.shape)

    x = torch.cat([x, skip], dim=1)
    print("After concatenation:", x.shape)

    x = model.refine(x)
    print("After refine:", x.shape)

    x = model.attention(x)
    print("After attention:", x.shape)

    out = model.conv(x)
    print("After final conv:", out.shape)

    macs, params = profile(model, inputs=(inp, skip))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"MACs: {macs}, Params: {params}")

def test_scattention():
    model = SCAttention(64).to('cuda')
    inp = torch.randn(1, 64, 128, 128).to('cuda')
    print(f"/n scattention")
    c_att = model.channel_gate(inp)
    print("After channel_gate:", c_att.shape)

    c_att = model.layer_norm(c_att)
    print("After layer_norm:", c_att.shape)

    channel_enhanced = inp * c_att.sigmoid()
    print("After channel enhancement:", channel_enhanced.shape)

    avg_out = torch.mean(inp, dim=1, keepdim=True)
    max_out = torch.amax(inp, dim=1, keepdim=True)
    s_in = torch.cat([avg_out, max_out], dim=1)
    print("After spatial input preparation:", s_in.shape)

    s_att = model.spatial_gate(s_in)
    print("After spatial_gate:", s_att.shape)

    spatial_enhanced = inp * s_att
    print("After spatial enhancement:", spatial_enhanced.shape)

    edge_enhanced = model.edge_conv(inp)
    print("After edge_conv:", edge_enhanced.shape)

    combined = torch.cat([channel_enhanced, spatial_enhanced], dim=1)
    print("After combining channel and spatial enhancements:", combined.shape)

    fused = model.fusion(combined) + edge_enhanced
    print("After fusion:", fused.shape)

    macs, params = profile(model, inputs=(inp,))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"MACs: {macs}, Params: {params}")

def test_scedgedetectionmodule():
    model = SCEdgeDetectionModule(64).to('cuda')
    inp = torch.randn(1, 64, 128, 128).to('cuda')
    print(f"/n scedgedetectionmodule")
    cdc_out = model.cdc(inp)
    print("After cdc:", cdc_out.shape)

    hdc_out = model.hdc(inp)
    print("After hdc:", hdc_out.shape)

    vdc_out = model.vdc(inp)
    print("After vdc:", vdc_out.shape)

    edge_feats = torch.cat([cdc_out, hdc_out, vdc_out], dim=1)
    print("After concatenation:", edge_feats.shape)

    fused = model.fusion(edge_feats)
    print("After fusion:", fused.shape)

    macs, params = profile(model, inputs=(inp,))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"MACs: {macs}, Params: {params}")

if __name__ == '__main__':
    test_scbackbone()
    # test_scencoderblock()
    # # test_scdecoderblock()
    # test_scattention()
    # test_scedgedetectionmodule()
