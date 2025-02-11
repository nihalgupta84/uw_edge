import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

###############################################
# 1. Data Transform Functions
###############################################
def data_transform(x):
    """
    Normalize input image to range [-1, 1].
    Mathematical operation: y = 2*x - 1.
    """
    return 2 * x - 1.0

def inverse_data_transform(x):
    """
    Inverse the normalization to restore pixel range to [0, 1].
    Mathematical operation: y = clamp((x + 1)/2, 0, 1).
    """
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

###############################################
# 2. Wavelet Transforms (DWT and IWT)
###############################################
def dwt_init(x):
    """
    Discrete Wavelet Transform (DWT) using Haar wavelets.
    The image is split into four subbands:
      - x_LL: low-frequency (approximation)
      - x_HL, x_LH, x_HH: high-frequency (details)
    These are computed by combining downsampled rows and columns.
    The function concatenates the four subbands along the batch dimension.
    """
    # Split along height: even and odd rows, scaled by 1/2.
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    # Now split each into even and odd columns.
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    # Compute the subbands.
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    # Concatenate along batch dimension.
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)

def iwt_init(x):
    """
    Inverse Wavelet Transform (IWT) using Haar wavelets.
    Reconstructs the image from four subbands that are concatenated along the batch dimension.
    Note: It assumes that the batch dimension is a multiple of 4.
    """
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch = in_batch // (r ** 2)
    out_channel = in_channel
    out_height = r * in_height
    out_width = r * in_width
    # Ensure proper slicing along all dimensions.
    x1 = x[0:out_batch, :, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width], dtype=x.dtype, device=x.device)
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h

class DWT(nn.Module):
    """
    Wrapper for the discrete wavelet transform.
    Note: No gradient is needed for these signal processing operations.
    """
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    """
    Wrapper for the inverse wavelet transform.
    """
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

###############################################
# 3. High-Frequency Residual Module (HFRM) and Sub-Modules
###############################################
class Depth_conv(nn.Module):
    """
    Depthwise separable convolution.
    It first applies a depthwise convolution and then a pointwise convolution.
    This is efficient and reduces the number of parameters.
    """
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class cross_attention(nn.Module):
    """
    A cross-attention module that computes attention between a query and a context.
    This allows the model to fuse information between different feature maps.
    """
    def __init__(self, dim, num_heads, dropout=0.):
        super(cross_attention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = dim // num_heads

        self.query = Depth_conv(in_ch=dim, out_ch=dim)
        self.key = Depth_conv(in_ch=dim, out_ch=dim)
        self.value = Depth_conv(in_ch=dim, out_ch=dim)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        # Rearrange tensor dimensions for multi-head attention.
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, ctx):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(ctx)
        mixed_value_layer = self.value(ctx)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Compute scaled dot-product attention.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        ctx_layer = torch.matmul(attention_probs, value_layer)
        ctx_layer = ctx_layer.permute(0, 2, 1, 3).contiguous()
        return ctx_layer

class Dilated_Resblock(nn.Module):
    """
    A dilated residual block that uses convolutions with different dilation rates.
    The dilations allow for a larger receptive field without increasing the number of parameters dramatically.
    """
    def __init__(self, in_channels, out_channels):
        super(Dilated_Resblock, self).__init__()
        sequence = []
        sequence += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                      padding=1, dilation=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                      padding=2, dilation=2),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1,
                      padding=3, dilation=3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1,
                      padding=2, dilation=2),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1,
                      padding=1, dilation=1)
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x) + x

class HFRM(nn.Module):
    """
    High-Frequency Residual Module:
    This module processes high-frequency components.
    It splits the input (assumed to have been concatenated along the batch dimension)
    into three parts, applies cross-attention between different splits, and fuses them.
    Finally, a residual connection is added.
    """
    def __init__(self, in_channels, out_channels):
        super(HFRM, self).__init__()
        self.conv_head = Depth_conv(in_channels, out_channels)

        self.dilated_block_LH = Dilated_Resblock(out_channels, out_channels)
        self.dilated_block_HL = Dilated_Resblock(out_channels, out_channels)

        self.cross_attention0 = cross_attention(out_channels, num_heads=8)
        self.dilated_block_HH = Dilated_Resblock(out_channels, out_channels)
        self.conv_HH = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1)
        self.cross_attention1 = cross_attention(out_channels, num_heads=8)

        self.conv_tail = Depth_conv(out_channels, in_channels)

    def forward(self, x):
        # x is expected to have batch size 3*n (concatenation of HL, LH, HH)
        b, c, h, w = x.shape
        residual = x

        x = self.conv_head(x)
        # Split x into three equal parts along the batch dimension.
        x_HL = x[:b // 3, ...]
        x_LH = x[b // 3: 2 * b // 3, ...]
        x_HH = x[2 * b // 3:, ...]

        # Compute cross-attention between the two splits.
        x_HH_LH = self.cross_attention0(x_LH, x_HH)
        x_HH_HL = self.cross_attention1(x_HL, x_HH)

        # Process HL and LH parts with dilated residual blocks.
        x_HL = self.dilated_block_HL(x_HL)
        x_LH = self.dilated_block_LH(x_LH)
        # Fuse the high-frequency part.
        x_HH = self.dilated_block_HH(self.conv_HH(torch.cat((x_HH_LH, x_HH_HL), dim=1)))

        # Concatenate the processed splits and apply a final convolution.
        out = self.conv_tail(torch.cat((x_HL, x_LH, x_HH), dim=0))
        return out + residual

###############################################
# 4. Restoration Model (Without Diffusion)
###############################################
class RestorationModel(nn.Module):
    """
    A model for image restoration that does not use the diffusion process.
    It employs a two-stage wavelet decomposition to separate low and high frequency components,
    enhances the high-frequency details with HFRM modules, and then reconstructs the image
    using the inverse wavelet transform.
    
    Mathematically, if I denote DWT and IWT as the wavelet and inverse wavelet transforms,
    and HFRM as an enhancement function, then the restoration can be written as:
    
         x_norm = data_transform(x)
         [LL, HF0] = DWT(x_norm)   [with HF0 containing HL, LH, HH]
         HF0_enh = HFRM0(HF0)
         [LL_LL, HF1] = DWT(LL)
         HF1_enh = HFRM1(HF1)
         recon_LL = IWT( [LL_LL, HF1_enh] )
         recon_img = IWT( [recon_LL, HF0_enh] )
         output = inverse_data_transform(recon_img)
    """
    def __init__(self):
        super(RestorationModel, self).__init__()
        # Two high-frequency enhancement modules.
        self.high_enhance0 = HFRM(in_channels=3, out_channels=64)
        self.high_enhance1 = HFRM(in_channels=3, out_channels=64)
        # Wavelet transforms.
        self.dwt = DWT()
        self.iwt = IWT()

    def forward(self, x):
        # x is assumed to be a tensor of shape (n, c, h, w).
        # 1. Normalize the input.
        input_img_norm = data_transform(x)
        # 2. First stage wavelet decomposition.
        x_dwt = self.dwt(input_img_norm)  # Output shape: (4*n, c, h/2, w/2)
        n = x.size(0)
        # Split the first quarter (LL) and the remaining (high frequency parts).
        x_LL = x_dwt[:n, ...]          # Low-frequency approximation.
        x_high0 = x_dwt[n:, ...]       # High-frequency details (should be 3*n in batch).
        
        # Enhance the first high-frequency branch.
        enhanced_high0 = self.high_enhance0(x_high0)
        
        # 3. Second stage on the low-frequency part.
        x_LL_dwt = self.dwt(x_LL)      # Output shape: (4*n, c, h/4, w/4)
        x_LL_LL = x_LL_dwt[:n, ...]     # Further low-frequency component.
        x_high1 = x_LL_dwt[n:, ...]     # High-frequency details from the LL component.
        enhanced_high1 = self.high_enhance1(x_high1)
        
        # 4. Reconstruction: first reconstruct the low-frequency branch.
        recon_LL = self.iwt(torch.cat((x_LL_LL, enhanced_high1), dim=0))
        # Then reconstruct the final restored image by combining with the first high-frequency branch.
        recon_img = self.iwt(torch.cat((recon_LL, enhanced_high0), dim=0))
        
        # 5. Inverse normalization to bring the output back to [0,1].
        output = inverse_data_transform(recon_img)
        return output

###############################################
# 5. Example Usage
###############################################
if __name__ == "__main__":
    # Create a dummy input image tensor with shape (n, c, h, w)
    # For example, n=1, c=3, h=256, w=256.
    dummy_input = torch.rand(1, 3, 256, 256)
    
    # Instantiate the restoration model.
    model = RestorationModel()

    # Forward pass: obtain the restored image.
    output = model(dummy_input)
    print("Output shape:", output.shape)
