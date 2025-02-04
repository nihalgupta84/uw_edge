import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

###############################################################################
# 1) ReflectionPaddingConv
###############################################################################
# PURPOSE:
#   - This utility module wraps a reflection pad + Conv2d operation.
#   - Reflection padding helps reduce boundary artifacts (especially in
#     underwater images).
#   - The shape flow for an input (B, C, H, W):
#       -> pad -> (B, C, H+2pad, W+2pad) -> conv -> (B, out_ch, H, W)
#
# USAGE:
#   Typically used in place of "Conv2d(..., padding=1)" but with reflection.
###############################################################################
class ReflectionPaddingConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1, bias=True):
        """
        Args:
            in_ch (int):    Number of input channels.
            out_ch (int):   Number of output channels.
            kernel_size (int): Convolution kernel size (3 by default).
            stride (int):   Convolution stride (1 by default).
            groups (int):   Group parameter for conv, used for depthwise if needed.
            bias (bool):    Whether to include conv bias.
        """
        super().__init__()
        # For kernel_size=3, pad =1. That means 1 reflection pad around each boundary.
        pad = (kernel_size - 1) // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=stride, padding=0, groups=groups, bias=bias)

    def forward(self, x):
        """
        Shape Flow:
          x: (B, in_ch, H, W)
          pad -> (B, in_ch, H+2*pad, W+2*pad)
          conv->(B, out_ch, H, W) [assuming no stride > 1]
        """
        x = self.pad(x)
        x = self.conv(x)
        return x


###############################################################################
# 2) SCEdgeDetectionModule
###############################################################################
# PURPOSE:
#   - This module applies THREE separate depthwise convolutions with
#     "edge" kernels (center-diff, Sobel horizontal, Sobel vertical).
#   - Then it fuses them via a 1x1 conv + GELU.
#   - Minimizes normalization so we don't "wash out" edges or over-normalize.
#
# SHAPE FLOW:
#    Input: (B, C, H, W)
#    Depthwise conv (C->C) done 3 times => cdc_out, hdc_out, vdc_out each (B, C, H, W).
#    Concatenate => (B, 3C, H, W).
#    1x1 conv => (B, C, H, W).
###############################################################################
class SCEdgeDetectionModule(nn.Module):
    def __init__(self, channels):
        """
        Args:
            channels (int): number of feature channels to detect edges for
        """
        super(SCEdgeDetectionModule, self).__init__()

        # Depthwise conv for each kernel
        # groups=channels => each channel is convolved independently
        self.cdc = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                             groups=channels, bias=False)
        self.hdc = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                             groups=channels, bias=False)
        self.vdc = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                             groups=channels, bias=False)

        # Fuse edges from the 3 kernels
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.GELU()
        )

        self._init_edge_kernels()

    def _init_edge_kernels(self):
        """
        We define:
          - cdc_kernel: center-diff approach
          - hdc_kernel: horizontal sobel
          - vdc_kernel: vertical sobel
        and set them as "buffers" for the depthwise conv weights.
        """
        # Center difference kernel
        cdc_kernel = torch.zeros(1,1,3,3)
        cdc_kernel[0,0,1,1] = 1
        cdc_kernel[0,0,:,:] -= 1/8
        epsilon = 1e-5
        cdc_kernel = cdc_kernel / (cdc_kernel.abs().sum() + epsilon)

        # Sobel horizontal
        hdc_kernel = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]).float().view(1,1,3,3) / 8.0

        # Sobel vertical
        vdc_kernel = torch.tensor([
            [-1,-2,-1],
            [ 0, 0, 0],
            [ 1, 2, 1]
        ]).float().view(1,1,3,3) / 8.0

        # Register as buffers so they move with device
        self.register_buffer('cdc_kernel', cdc_kernel)
        self.register_buffer('hdc_kernel', hdc_kernel)
        self.register_buffer('vdc_kernel', vdc_kernel)

    def forward(self, x):
        """
        Steps:
          1) We replicate the custom kernels across 'C' channels (since groups=C).
          2) Depthwise conv with each kernel -> cdc_out, hdc_out, vdc_out
          3) Concatenate => fuse => return
        Shape:
          x: (B, C, H, W)
          out: (B, C, H, W)
        """
        B,C,H,W = x.shape

        # Replicate the custom kernels along channel dimension
        cdc_w = self.cdc_kernel.repeat(C,1,1,1)  # shape(C,1,3,3)
        hdc_w = self.hdc_kernel.repeat(C,1,1,1)
        vdc_w = self.vdc_kernel.repeat(C,1,1,1)

        # Overwrite the weight data (ensuring the conv uses these kernels)
        self.cdc.weight.data = cdc_w
        self.hdc.weight.data = hdc_w
        self.vdc.weight.data = vdc_w

        cdc_out = self.cdc(x)
        hdc_out = self.hdc(x)
        vdc_out = self.vdc(x)

        edge_feats = torch.cat([cdc_out, hdc_out, vdc_out], dim=1) # (B,3C,H,W)
        fused = self.fusion(edge_feats)                            # (B,C,H,W)

        return fused


###############################################################################
# 3) SCAttention (Simplified)
###############################################################################
# PURPOSE:
#   - Perform channel + spatial gating on the input features
#   - Minimizes instance norms or dropout to reduce risk of "over-smoothing"
#
# SHAPE FLOW:
#   Input: (B, C, H, W)
#   Channel: 
#     - global avg pool => (B, C, 1, 1)
#     - small MLP => (B, C, 1, 1)
#     - out => channel mask in [0,1]
#   Spatial:
#     - avg_out => (B,1,H,W)
#     - max_out => (B,1,H,W)
#     - concat =>(B,2,H,W)
#     - conv =>(B,1,H,W)
#   Then fuse => out =>(B,C,H,W)
###############################################################################
class SCAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        """
        Args:
            channels (int): number of channels
            reduction (int): channel gating factor
        """
        super().__init__()
        # Channel gate: uses 2-layer MLP with GELU. Finally a Sigmoid => (B,C,1,1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )

        # Spatial gate: concatenates avg+max => (B,2,H,W), conv => (B,1,H,W)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # final fuse is 1x1 conv
        self.fuse = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        """
        1) channel attention => x_ca
        2) spatial attention => x_sa
        3) sum => fuse => out
        shape: x => (B,C,H,W) => out => (B,C,H,W)
        """
        B, C, H, W = x.shape

        # Channel branch
        c_att = self.channel_gate(x)    # (B,C,1,1)
        x_ca = x * c_att                # broadcast along H,W

        # Spatial branch
        avg_out = torch.mean(x, dim=1, keepdim=True)       # (B,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)     # (B,1,H,W)
        s_in = torch.cat([avg_out, max_out], dim=1)        # (B,2,H,W)
        s_att = self.spatial_gate(s_in)                    # (B,1,H,W)
        x_sa = x * s_att

        # fuse channel & spatial 
        out = x_ca + x_sa
        out = self.fuse(out)
        return out


###############################################################################
# 4) SCEncoderBlock
###############################################################################
# PURPOSE:
#   - A single "encoder block" that:
#       1) does reflection conv -> out
#       2) runs out through edge detection & attention
#       3) sums them up
#       4) minimal dropout
#       5) second reflection conv
#       6) add skip connection if channels differ
#
# SHAPE FLOW:
#   Input (B, in_ch, H, W)
#   conv1 => (B, out_ch, H, W)
#   edge_detect & attention => each => (B, out_ch, H, W)
#   sum => dropout => conv2 => skip =>(B, out_ch, H, W)
###############################################################################
class SCEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        """
        Args:
            in_ch  (int): input channels
            out_ch (int): output channels
        """
        super().__init__()

        # 1) reflection conv + gelu
        self.conv1 = nn.Sequential(
            ReflectionPaddingConv(in_ch, out_ch, kernel_size=3, stride=1),
            nn.GELU()
        )

        # 2) edge detection + attention
        self.edge_detect = SCEdgeDetectionModule(out_ch)
        self.attention   = SCAttention(out_ch)

        # 3) second reflection conv
        self.conv2 = nn.Sequential(
            ReflectionPaddingConv(out_ch, out_ch, kernel_size=3),
            nn.GELU()
        )

        # skip conv if channel changes
        if in_ch != out_ch:
            self.skip_conv = ReflectionPaddingConv(in_ch, out_ch, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()

        # minimal dropout
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        """
        Step by step:
          1) out = conv1(x)
          2) e_feats = edge_detect(out)
             a_feats = attention(out)
             out = out + e_feats + a_feats
          3) out = dropout(out)
          4) out = conv2(out)
          5) skip = skip_conv(x)
          6) out = out + skip
        """
        out = self.conv1(x)

        e_feats = self.edge_detect(out)
        a_feats = self.attention(out)
        out = out + e_feats + a_feats

        out = self.dropout(out)
        out = self.conv2(out)

        skip_conn = self.skip_conv(x)
        out = out + skip_conn

        return out


###############################################################################
# 5) SCDecoderBlock
###############################################################################
# PURPOSE:
#   - Mirror of encoder, but we first upsample, conv, then fuse skip,
#     refine, attention, final 2x conv.
# SHAPE FLOW:
#   Input x =>(B, in_ch, H, W)
#   upsample =>(B, in_ch, 2H, 2W)
#   conv_up =>(B, in_ch, 2H, 2W)
#   concat skip =>(B, in_ch+skip_ch, 2H, 2W)
#   refine =>(B, in_ch+skip_ch, 2H, 2W)
#   attention =>(B, in_ch+skip_ch, 2H, 2W)
#   conv_final =>(B, out_ch, 2H, 2W)
###############################################################################
class SCDecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        """
        Args:
            in_ch   (int): channels from the lower-level (bottleneck) stage
            skip_ch (int): channels from the skip connection
            out_ch  (int): desired output channels for this decoder
        """
        super().__init__()
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # reflection conv after upsample
        self.conv_up = nn.Sequential(
            ReflectionPaddingConv(in_ch, in_ch, kernel_size=3),
            nn.GELU()
        )

        # refine
        self.refine = nn.Sequential(
            ReflectionPaddingConv(in_ch + skip_ch, in_ch + skip_ch, kernel_size=1),
            nn.GELU()
        )

        # simple attention
        self.attention = SCAttention(in_ch + skip_ch)

        # final 2-layer conv
        self.conv_final = nn.Sequential(
            ReflectionPaddingConv(in_ch + skip_ch, out_ch, kernel_size=3),
            nn.GELU(),
            ReflectionPaddingConv(out_ch, out_ch, kernel_size=3),
            nn.GELU()
        )

    def forward(self, x, skip):
        """
        Step by step:
          1) x_up = upsample(x)
          2) x_up = conv_up(x_up)
          3) out = cat([x_up, skip], dim=1)
          4) out = refine(out)
          5) out = attention(out)
          6) out = conv_final(out)
        """
        x_up = self.upsample(x)               # (B, in_ch, 2H, 2W)
        x_up = self.conv_up(x_up)             # (B, in_ch, 2H, 2W)

        out = torch.cat([x_up, skip], dim=1)  # (B, in_ch+skip_ch, 2H, 2W)
        out = self.refine(out)                # (B, in_ch+skip_ch, 2H, 2W)
        out = self.attention(out)             # (B, in_ch+skip_ch, 2H, 2W)
        out = self.conv_final(out)            # (B, out_ch, 2H, 2W)
        return out


###############################################################################
# 6) SCBackbone (Refined)
###############################################################################
# PURPOSE:
#   - A U-Net style backbone with minimal instance norm usage,
#     reflection padding throughout, simpler edge detection + attention,
#     lower dropout => helps preserve structure => can help SSIM.
#
# SHAPE FLOW (assuming base_ch=64):
#   1) init_conv:   (B,3,H,W)->(B,64,H,W)
#   2) encoder1 + down1 =>(B,64,H/2,W/2)
#   3) encoder2 + down2 =>(B,128,H/4,W/4)
#   4) encoder3 + down3 =>(B,256,H/8,W/8)
#   5) bottleneck =>(B,512,H/8,W/8)
#   6) decoder3 =>(B,256,H/4,W/4)
#   7) decoder2 =>(B,128,H/2,W/2)
#   8) decoder1 =>(B,64,H,W)
#   9) final =>(B,3,H,W)
###############################################################################
class SCBackbone(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        """
        Args:
            in_ch  (int): Input image channels, typically 3 for RGB.
            base_ch(int): The base number of channels (64 is default).
        """
        super().__init__()

        # Init: two conv steps with reflection
        self.init_conv = nn.Sequential(
            ReflectionPaddingConv(in_ch, base_ch, kernel_size=7),
            nn.GELU(),
            ReflectionPaddingConv(base_ch, base_ch, kernel_size=3),
            nn.GELU()
        )

        # encoders
        self.encoder1 = SCEncoderBlock(base_ch, base_ch)
        self.down1 = nn.MaxPool2d(2)  # =>(B,base_ch,H/2,W/2)

        self.encoder2 = SCEncoderBlock(base_ch, base_ch*2)
        self.down2 = nn.MaxPool2d(2) # =>(B,2*base_ch,H/4,W/4)

        self.encoder3 = SCEncoderBlock(base_ch*2, base_ch*4)
        self.down3 = nn.MaxPool2d(2) # =>(B,4*base_ch,H/8,W/8)

        # bottleneck
        self.bottleneck = SCEncoderBlock(base_ch*4, base_ch*8)
        # =>(B,8*base_ch,H/8,W/8) same resolution as after down3

        # decoders
        self.decoder3 = SCDecoderBlock(base_ch*8, base_ch*4, base_ch*4)
        self.decoder2 = SCDecoderBlock(base_ch*4, base_ch*2, base_ch*2)
        self.decoder1 = SCDecoderBlock(base_ch*2, base_ch, base_ch)

        # final
        # reflection conv =>(B, base_ch, H, W)
        # reflection conv =>(B, in_ch, H, W)
        # Sigmoid => force [0,1]
        self.final = nn.Sequential(
            ReflectionPaddingConv(base_ch, base_ch, kernel_size=3),
            nn.GELU(),
            ReflectionPaddingConv(base_ch, in_ch, kernel_size=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Example shape flow if input is (B,3,H,W):

          1) x0 = init_conv(x) =>(B,base_ch,H,W)
          2) s1 = encoder1(x0)->(B,base_ch,H,W)
             x1= down1(s1)   ->(B,base_ch,H/2,W/2)

          3) s2 = encoder2(x1)->(B,2*base_ch,H/2,W/2)
             x2= down2(s2)   ->(B,2*base_ch,H/4,W/4)

          4) s3 = encoder3(x2)->(B,4*base_ch,H/4,W/4)
             x3= down3(s3)   ->(B,4*base_ch,H/8,W/8)

          5) b = bottleneck(x3)->(B,8*base_ch,H/8,W/8)

          6) d3= decoder3(b,s3)->(B,4*base_ch,H/4,W/4)
          7) d2= decoder2(d3,s2)->(B,2*base_ch,H/2,W/2)
          8) d1= decoder1(d2,s1)->(B,base_ch,H,W)

          9) out= final(d1) ->(B,in_ch,H,W)
        """
        # encoder path
        x0 = self.init_conv(x)

        s1 = self.encoder1(x0)
        x1 = self.down1(s1)

        s2 = self.encoder2(x1)
        x2 = self.down2(s2)

        s3 = self.encoder3(x2)
        x3 = self.down3(s3)

        # bottleneck
        b  = self.bottleneck(x3)

        # decoder path
        d3 = self.decoder3(b, s3)
        d2 = self.decoder2(d3, s2)
        d1 = self.decoder1(d2, s1)

        # final conv => [0,1]
        out = self.final(d1)

        # Return final features (d1) if desired, and the final image
        return d1, out
    

class EdgeModel_V2(nn.Module):
    """
    Upgraded 'ScaterringBranch' that uses our new SCBackbone
    for advanced haze removal + scattering correction.

    The forward(...) MUST return: (featrueHR, hazeRemoval)
    to maintain compatibility with HRNet.py usage:
        featrueHR, hazeRemoval = self.hfBranch(input)
    """
    def __init__(self, in_channels=3, base_channels=64):
        super(EdgeModel_V2, self).__init__()
        # Instantiating our new backbone
        self.ch_in = 3
        self.down_depth = 2        
        self.backbone = SCBackbone(in_ch=self.ch_in, base_ch=64)

    def forward(self, input):
        """
        input: (B, 3, H, W)  # if you're using RGB
        returns: (featrueHR, hazeRemoval)
        """
        featrueHR, hazeRemoval = self.backbone(input)
        out = hazeRemoval
        return out
    
    def _get_featurehr(self, input):
        """
        input: (B, 3, H, W)  # if you're using RGB 
        returns: (featrueHR)
        """
        featrueHR, hazeRemoval = self.backbone(input)
        return featrueHR


def test_model():
    model = EdgeModel().to('cuda')
    inp = torch.randn(1, 3, 256, 256).to('cuda')
    print("Model Summary:")
    summary(model, input_size=(1, 3, 256, 256))

    out = model(inp)
    print(out.shape)

if __name__ == '__main__':
    from thop import profile
    from thop import clever_format
    from torchinfo import summary
    inp = torch.randn(1, 3, 256, 256)
    model = EdgeModel()
    macs, params = profile(model, inputs=(inp,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
    test_model()