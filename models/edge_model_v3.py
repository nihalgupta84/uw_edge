import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import MAQ, QBlock, Aggreation, SPP, ColorBalancePrior
from torchinfo import summary
###############################################################################
# 1) ReflectionPatchEmbed
###############################################################################
# Replaces or modifies the SOTA OverlapPatchEmbed to use reflection padding
# before the conv. Typically we do stride=1 so that output resolution is 
# the same as input. If the SOTA uses stride=2 or some partial overlap, 
# adapt accordingly.
###############################################################################
class ReflectionPatchEmbed(nn.Module):
    """
    Reflection-based patch embedding layer.
    Replaces OverlapPatchEmbed from the SOTA code. 
    Possibly beneficial to reduce corner artifacts in underwater images.
    """
    def __init__(self, in_ch=3, embed_dim=48, kernel_size=3, stride=1, bias=False):
        super().__init__()
        pad = (kernel_size - 1)//2
        self.reflection = nn.ReflectionPad2d(pad)
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=kernel_size, 
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        """
        x: (B, in_ch, H, W)
        reflection -> (B, in_ch, H+2pad, W+2pad)
        conv       -> (B, embed_dim, H/stride, W/stride)
        """
        x = self.reflection(x)
        x = self.proj(x)
        return x


###############################################################################
# 2) SCEdgeDetectionModule (Optional)
###############################################################################
# Your prior edge detection approach, kept simpler.
###############################################################################
class SCEdgeDetectionModule(nn.Module):
    """
    Depthwise edge detection with 3 custom kernels (CDC, Sobel-H, Sobel-V).
    Then a 1x1 conv fuse with a small nonlinear. 
    You can apply it once at the beginning or skip altogether.
    """
    def __init__(self, channels):
        super(SCEdgeDetectionModule, self).__init__()
        # Depthwise conv for each kernel
        self.cdc = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.hdc = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.vdc = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

        # fuse
        self.fuse = nn.Sequential(
            nn.Conv2d(channels*3, channels, 1),
            nn.GELU()
        )
        self._init_kernels()

    def _init_kernels(self):
        # Could define your center-diff, sobel-h, sobel-v as before
        # This is just an example:
        import numpy as np

        cdc_kernel = torch.zeros(1,1,3,3)
        cdc_kernel[0,0,1,1] = 1
        cdc_kernel[0,0,:,:] -= 1/8
        cdc_kernel /= cdc_kernel.abs().sum() + 1e-5

        hdc_kernel = torch.tensor([
            [-1,0,1],
            [-2,0,2],
            [-1,0,1]
        ]).float().view(1,1,3,3)/8.0

        vdc_kernel = torch.tensor([
            [-1,-2,-1],
            [0,0,0],
            [1,2,1]
        ]).float().view(1,1,3,3)/8.0

        self.register_buffer('cdc_k', cdc_kernel)
        self.register_buffer('hdc_k', hdc_kernel)
        self.register_buffer('vdc_k', vdc_kernel)

    def forward(self, x):
        B,C,H,W = x.shape
        # replicate the kernels
        cdc_w = self.cdc_k.repeat(C,1,1,1)
        hdc_w = self.hdc_k.repeat(C,1,1,1)
        vdc_w = self.vdc_k.repeat(C,1,1,1)

        self.cdc.weight.data = cdc_w
        self.hdc.weight.data = hdc_w
        self.vdc.weight.data = vdc_w

        c_out = self.cdc(x)
        h_out = self.hdc(x)
        v_out = self.vdc(x)
        cat = torch.cat([c_out, h_out, v_out], dim=1)
        fused = self.fuse(cat)
        return fused


###############################################################################
# 3) Example MAQ / QBlock / Aggreation / SPP from SOTA
###############################################################################
# We'll show placeholders. In your real code, you import from the SOTA files:
# e.g., "from .model import MAQ, QBlock, Aggreation, SPP" 
###############################################################################
# class MAQ(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         # We'll do a minimal pass. In reality, use the real SOTA code.
#         self.dummy = nn.Conv2d(dim, dim, 3, padding=1)
#     def forward(self, x, prior):
#         return self.dummy(x)

# class QBlock(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dummy = nn.Conv2d(dim, dim, 3, padding=1)
#     def forward(self, x):
#         return self.dummy(x)

# class Aggreation(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.dummy = nn.Conv2d(in_ch, out_ch, 1)
#     def forward(self, x):
#         return self.dummy(x)

# class SPP(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
#     def forward(self, x):
#         return self.conv(x)


###############################################################################
# 4) ReflectionFeatureContextualizer
###############################################################################
# Replaces FeatureContextualizer from SOTA with reflection-based embed,
# optional single edge detection, then same MAQ blocks + aggregator + SPP.
###############################################################################
class ReflectionFeatureContextualizer(nn.Module):
    """
    Suppose SOTA used:
      - OverlapPatchEmbed(ch_in, dim)
      - block1_1 = MAQ(dim), block1_2=MAQ(dim), aggregator => x1
      - block2_1=MAQ(dim), block2_2=MAQ(dim), aggregator => x2
      - final SPP => out
    We replicate the same structure, but replace OverlapPatchEmbed with 
    ReflectionPatchEmbed, plus optional edge detection after embed.
    """
    def __init__(self, ch_in=3, dim=48, ch_out=6, use_edge=True):
        super().__init__()
        self.use_edge = use_edge
        self.embed = ReflectionPatchEmbed(in_ch=ch_in, embed_dim=dim, kernel_size=3, stride=1, bias=False)
        # If you want a second embed for 'prior', or same
        self.embed_prior = ReflectionPatchEmbed(in_ch=ch_in, embed_dim=dim, kernel_size=3, stride=1, bias=False)

        if use_edge:
            self.edge = SCEdgeDetectionModule(dim)

        # SOTA approach
        self.block1_1 = MAQ(dim)
        self.block1_2 = MAQ(dim)
        self.agg1 = Aggreation(dim*2, dim)

        self.block2_1 = MAQ(dim)
        self.block2_2 = MAQ(dim)
        self.agg2 = Aggreation(dim*3, dim)

        self.spp = SPP(dim, ch_out)

    def forward(self, x, prior):
        # 1) reflection embed
        x = self.embed(x)              # => (B, dim, H, W)
        prior_embed = self.embed_prior(prior)

        # 2) optional single edge detection
        if self.use_edge:
            x = self.edge(x)

        # 3) block1
        x_1 = self.block1_1(x, prior_embed)
        x_2 = self.block1_2(x_1, x_1)
        x1 = self.agg1(torch.cat([x_1, x_2], dim=1))

        # 4) block2
        x_1 = self.block2_1(x1, prior_embed)
        x_2 = self.block2_2(x_1, x_1)
        x2 = self.agg2(torch.cat([x1, x_1, x_2], dim=1))

        # 5) SPP
        out = self.spp(x2)
        return out


###############################################################################
# 5) ReflectionDetailRestorer
###############################################################################
# Replaces SOTA's DetailRestorer with reflection-based embed + optional edge,
# then QBlock aggregator, SPP. 
###############################################################################
class ReflectionDetailRestorer(nn.Module):
    def __init__(self, ch=3, dim=16, use_edge=False):
        super().__init__()
        self.use_edge = use_edge
        self.embed = ReflectionPatchEmbed(in_ch=ch, embed_dim=dim, kernel_size=3, stride=1, bias=False)

        if use_edge:
            self.edge = SCEdgeDetectionModule(dim)

        self.block1_1 = QBlock(dim)
        self.block1_2 = QBlock(dim)
        self.agg1 = Aggreation(dim*2, dim)

        self.block2_1 = QBlock(dim)
        self.block2_2 = QBlock(dim)
        self.agg2 = Aggreation(dim*3, dim)

        self.block3_1 = QBlock(dim)
        self.block3_2 = QBlock(dim)
        self.agg3 = Aggreation(dim*4, dim)

        self.spp = SPP(dim, ch)

    def forward(self, x):
        # embed
        x = self.embed(x)
        if self.use_edge:
            x = self.edge(x)

        x_1 = self.block1_1(x)
        x_2 = self.block1_2(x_1)
        x1 = self.agg1(torch.cat([x_1, x_2], dim=1))

        x_1 = self.block2_1(x1)
        x_2 = self.block2_2(x_1)
        x2 = self.agg2(torch.cat([x1, x_1, x_2], dim=1))

        x_1 = self.block3_1(x2)
        x_2 = self.block3_2(x_1)
        x3 = self.agg3(torch.cat([x1, x2, x_1, x_2], dim=1))

        out = self.spp(x3)
        return out

def check_nan(x, msg):
    """Utility to check for NaNs and print a message. 
       You can comment this out if everything is stable."""
    if torch.isnan(x).any():
        print(f"[NaN Alert] in {msg}")
    return x

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
###############################################################################
# 6) ProposedReflectionSOTA Model
###############################################################################
# Here we demonstrate how you can unify it into a single model that
# uses a color prior, reflection-based FeatureContextualizer, reflection-based
# DetailRestorer, etc. The final training script can remain the same as SOTA.
###############################################################################
class EdgeModel_V3(nn.Module):
    def __init__(self, ch_in=3, dim_fc=48, ch_out_fc=6, dim_dr=16, use_edge=True):
        """
        Args:
          ch_in    : input channels
          dim_fc   : dimension for ReflectionFeatureContextualizer
          ch_out_fc: output channels from feature contextualizer
          dim_dr   : dimension for detail restorer
          color_prior: optional color prior module if the SOTA uses it
          use_edge : whether to do edge detection once in FC or DR
        """
        super().__init__()
        self.ch_in = ch_in
        self.color_prior = ColorBalancePrior(ch_in) # e.g., ColorBalancePrior(3) if you want

        self.fc = ReflectionFeatureContextualizer(ch_in=ch_in, dim=dim_fc, ch_out=ch_out_fc, use_edge=use_edge)
        self.dr = ReflectionDetailRestorer(ch=ch_in, dim=dim_dr, use_edge=use_edge)

        # final could be scale-harmonizer, or simply a conv + Sigmoid
        # for demonstration, do a single conv + Sigmoid
        self.final = nn.Sequential(
            nn.Conv2d(ch_in + ch_out_fc, ch_in, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.apply(init_weights)
    def forward(self, x):
        """
        1) optional color prior
        2) reflection-based feature contextualizer =>(B,ch_out_fc,H,W)
        3) reflection-based detail restorer =>(B,ch, H,W)
        4) final fusion =>(B,ch_in,H,W)
        """
        prior = check_nan(self.color_prior(x), "ColorPrior")
        # prior = self.color_prior(x)  # (B,3,H,W)
        fc_out = check_nan(self.fc(x, prior), "FeatureContextualizer")
        # fc_out = self.fc(x, prior)  # =>(B,ch_out_fc,H,W)
        dr_out = check_nan(self.dr(x), "DetailRestorer")
        # dr_out = self.dr(x)         # =>(B,ch_in,H,W)

        cat = torch.cat([fc_out, dr_out], dim=1)  # =>(B, ch_out_fc + ch_in, H,W)
        out = check_nan(self.final(cat), "FinalConv")
        # out = self.final(cat)                     # =>(B,ch_in,H,W)
        return out



def test_edge_model_v3():
    model = EdgeModel_V3(ch_in=3, dim_fc=48, ch_out_fc=6, dim_dr=16, use_edge=False)
    x = torch.randn(1, 3, 256, 256)  # Example input tensor
    output = model(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output mean:", output.mean().item())
    print("Output std:", output.std().item())
    
    # Print model summary
    summary(model, input_size=(1, 3, 256, 256))

if __name__ == "__main__":
    test_edge_model_v3()

3