#File: models/model.py (state of the art model)

import numbers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from numpy.random import RandomState
from scipy.stats import chi
from torchinfo import summary


class FeatureContextualizer(nn.Module):
    def __init__(self, ch_in=3, dim=16, ch_out=6):
        super(FeatureContextualizer, self).__init__()

        self.embed = OverlapPatchEmbed(ch_in, dim)
        self.embed_prior = OverlapPatchEmbed(ch_in, dim)

        self.block1_1 = MAQ(dim)
        self.block1_2 = MAQ(dim)
        self.agg1 = Aggreation(dim * 2, dim)

        self.block2_1 = MAQ(dim)
        self.block2_2 = MAQ(dim)
        self.agg2 = Aggreation(dim * 3, dim)

        self.spp = SPP(dim, ch_out)

    def forward(self, x, prior):
        x = self.embed(x)
        prior_embed = self.embed_prior(prior)

        x_1 = self.block1_1(x, prior_embed)
        x_2 = self.block1_2(x_1, x_1)
        x1 = self.agg1(torch.cat((x_1, x_2), dim=1))

        x_1 = self.block2_1(x1, prior_embed)
        x_2 = self.block2_2(x_1, x_1)
        x2 = self.agg2(torch.cat((x1, x_1, x_2), dim=1))

        out = self.spp(x2)

        return out


class DetailRestorer(nn.Module):
    def __init__(self, ch=3, dim=16):
        super(DetailRestorer, self).__init__()

        self.embed = OverlapPatchEmbed(ch, dim)

        self.block1_1 = QBlock(dim)
        self.block1_2 = QBlock(dim)
        self.agg1 = Aggreation(dim * 2, dim)

        self.block2_1 = QBlock(dim)
        self.block2_2 = QBlock(dim)
        self.agg2 = Aggreation(dim * 3, dim)

        self.block3_1 = QBlock(dim)
        self.block3_2 = QBlock(dim)
        self.agg3 = Aggreation(dim * 4, dim)

        self.spp = SPP(dim, ch)

    def forward(self, x):
        x = self.embed(x)

        x_1 = self.block1_1(x)
        x_2 = self.block1_2(x_1)
        x1 = self.agg1(torch.cat((x_1, x_2), dim=1))

        x_1 = self.block2_1(x1)
        x_2 = self.block2_2(x_1)
        x2 = self.agg2(torch.cat((x1, x_1, x_2), dim=1))

        x_1 = self.block3_1(x2)
        x_2 = self.block3_2(x_1)
        x3 = self.agg3(torch.cat((x1, x2, x_1, x_2), dim=1))

        out = self.spp(x3)

        return out


class Condition(nn.Module):
    def __init__(self, in_nc=3, nf=32):
        super(Condition, self).__init__()
        stride = 2
        pad = 0
        self.pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(in_nc, nf, 7, stride, pad, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.conv3 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1_out = self.act(self.conv1(self.pad(x)))
        conv2_out = self.act(self.conv2(self.pad(conv1_out)))
        conv3_out = self.act(self.conv3(self.pad(conv2_out)))
        out = torch.mean(conv3_out, dim=[2, 3], keepdim=False)

        return out


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim_in, dim_out, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim_in * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim_in, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim_out, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * num_heads * 2, kernel_size=1, bias=bias)
        self.q = nn.Conv2d(dim, dim * num_heads, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim * num_heads, dim * num_heads, kernel_size=3,
                                  stride=1, padding=1, groups=dim * num_heads, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * num_heads * 2, dim * num_heads * 2, kernel_size=3,
                                   stride=1, padding=1, groups=dim * num_heads * 2, bias=bias)
        self.project_out = nn.Conv2d(dim * num_heads, dim, kernel_size=1, bias=bias)

    def forward(self, q, k):
        b, c, h, w = k.shape

        kv = self.kv_dwconv(self.kv(k))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(q))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class CrossAttentionTransformer(nn.Module):
    def __init__(self, dim, num_heads=2, ffn_expansion_factor=2.66, bias=True, layerNorm_type='WithBias'):
        super(CrossAttentionTransformer, self).__init__()

        self.norm1 = LayerNorm(dim, layerNorm_type)
        self.norm2 = LayerNorm(dim, layerNorm_type)
        self.attn = CrossAttention(dim, num_heads, bias)

        self.norm3 = LayerNorm(dim, layerNorm_type)
        self.ffn = FeedForward(dim, dim, ffn_expansion_factor, bias)

    # query: x  key:y   value:y
    def forward(self, x, y):
        y = y + self.attn(self.norm1(x), self.norm2(y))
        y = y + self.ffn(self.norm3(y))

        return y


class SelfAttentionTransformer(nn.Module):
    def __init__(self, dim, num_heads=1, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(SelfAttentionTransformer, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class ContextBlock(nn.Module):

    def __init__(self, n_feat, bias=False):
        super(ContextBlock, self).__init__()

        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        )

    def modeling(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.modeling(x)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term

        return x


class ScaleHarmonizer(nn.Module):
    def __init__(self, in_nc=6, out_nc=3, base_nf=64, cond_nf=32):
        super(ScaleHarmonizer, self).__init__()

        self.base_nf = base_nf
        self.out_nc = out_nc

        self.cond_net = Condition(in_nc=in_nc, nf=cond_nf)

        self.cond_scale1 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_scale2 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_scale3 = nn.Linear(cond_nf, out_nc, bias=True)

        self.cond_shift1 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_shift2 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_shift3 = nn.Linear(cond_nf, out_nc, bias=True)

        self.conv1 = nn.Conv2d(in_nc, base_nf, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(base_nf, base_nf, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(base_nf, out_nc, 1, 1, bias=True)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        cond = self.cond_net(x)

        scale1 = self.cond_scale1(cond)
        shift1 = self.cond_shift1(cond)

        scale2 = self.cond_scale2(cond)
        shift2 = self.cond_shift2(cond)

        scale3 = self.cond_scale3(cond)
        shift3 = self.cond_shift3(cond)

        # Feature Calibrator
        out = self.conv1(x)
        out = out * scale1.view(-1, self.base_nf, 1, 1) + shift1.view(-1, self.base_nf, 1, 1) + out
        out = self.act(out)
        # Feature Calibrator
        out = self.conv2(out)
        out = out * scale2.view(-1, self.base_nf, 1, 1) + shift2.view(-1, self.base_nf, 1, 1) + out
        out = self.act(out)
        # Feature Calibrator
        out = self.conv3(out)
        out = out * scale3.view(-1, self.out_nc, 1, 1) + shift3.view(-1, self.out_nc, 1, 1) + out

        return out


# --------- Residual Context Block (RCB) ----------
class RCB(nn.Module):
    def __init__(self, n_feat, bias=False, groups=1):
        super(RCB, self).__init__()

        act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act,
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups)
        )

        self.act = act

        self.gcnet = ContextBlock(n_feat, bias=bias)

    def forward(self, x):
        res = self.body(x)
        res = self.act(self.gcnet(res))
        res += x
        return res


class QuaternionConv(nn.Module):
    r"""Applies a Quaternion Convolution to the incoming data.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilatation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='quaternion', seed=None, operation='convolution2d', rotation=True, quaternion_format=True,
                 scale=False):

        super(QuaternionConv, self).__init__()

        self.in_channels = in_channels // 4
        self.out_channels = out_channels // 4
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilatation = dilatation
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.operation = operation
        self.rotation = rotation
        self.quaternion_format = quaternion_format
        self.winit = {'quaternion': quaternion_init,
                      'unitary': unitary_init,
                      'random': random_init}[self.weight_init]
        self.scale = scale

        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape(self.operation,
                                                                       self.in_channels, self.out_channels, kernel_size)

        self.r_weight = nn.Parameter(torch.Tensor(*self.w_shape))
        self.i_weight = nn.Parameter(torch.Tensor(*self.w_shape))
        self.j_weight = nn.Parameter(torch.Tensor(*self.w_shape))
        self.k_weight = nn.Parameter(torch.Tensor(*self.w_shape))

        if self.scale:
            self.scale_param = nn.Parameter(torch.Tensor(self.r_weight.shape))
        else:
            self.scale_param = None

        if self.rotation:
            self.zero_kernel = nn.Parameter(torch.zeros(self.r_weight.shape), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                         self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, inp):

        if self.rotation:
            return quaternion_conv_rotation(inp, self.zero_kernel, self.r_weight, self.i_weight, self.j_weight,
                                            self.k_weight, self.bias, self.stride, self.padding, self.groups,
                                            self.dilatation,
                                            self.quaternion_format, self.scale_param)
        else:
            return quaternion_conv(inp, self.r_weight, self.i_weight, self.j_weight,
                                   self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels=' + str(self.in_channels) \
            + ', out_channels=' + str(self.out_channels) \
            + ', bias=' + str(self.bias is not None) \
            + ', kernel_size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', seed=' + str(self.seed) \
            + ', rotation=' + str(self.rotation) \
            + ', q_format=' + str(self.quaternion_format) \
            + ', operation=' + str(self.operation) + ')'


def quaternion_init(in_features, out_features, rng, kernel_size=None, criterion='glorot'):
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

    # Generating randoms and purely imaginary quaternions :
    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    modulus = chi.rvs(4, loc=0, scale=s, size=kernel_shape)
    number_of_weights = np.prod(kernel_shape)
    v_i = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_j = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_k = np.random.uniform(-1.0, 1.0, number_of_weights)

    # Purely imaginary quaternions unitary
    for i in range(0, number_of_weights):
        norm = np.sqrt(v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2 + 0.0001)
        v_i[i] /= norm
        v_j[i] /= norm
        v_k[i] /= norm
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

    weight_r = modulus * np.cos(phase)
    weight_i = modulus * v_i * np.sin(phase)
    weight_j = modulus * v_j * np.sin(phase)
    weight_k = modulus * v_k * np.sin(phase)

    return (weight_r, weight_i, weight_j, weight_k)


def unitary_init(in_features, out_features, rng, kernel_size=None, criterion='he'):
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in = in_features * receptive_field
        fan_out = out_features * receptive_field
    else:
        fan_in = in_features
        fan_out = out_features

    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    number_of_weights = np.prod(kernel_shape)
    v_r = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_i = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_j = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_k = np.random.uniform(-1.0, 1.0, number_of_weights)

    # Unitary quaternion
    for i in range(0, number_of_weights):
        norm = np.sqrt(v_r[i] ** 2 + v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2) + 0.0001
        v_r[i] /= norm
        v_i[i] /= norm
        v_j[i] /= norm
        v_k[i] /= norm
    v_r = v_r.reshape(kernel_shape)
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    return (v_r, v_i, v_j, v_k)


def affect_init_conv(r_weight, i_weight, j_weight, k_weight, kernel_size, init_func, rng,
                     init_criterion):
    if r_weight.size() != i_weight.size() or r_weight.size() != j_weight.size() or \
            r_weight.size() != k_weight.size():
        raise ValueError('The real and imaginary weights '
                         'should have the same size . Found: r:'
                         + str(r_weight.size()) + ' i:'
                         + str(i_weight.size()) + ' j:'
                         + str(j_weight.size()) + ' k:'
                         + str(k_weight.size()))

    elif 2 >= r_weight.dim():
        raise Exception('affect_conv_init accepts only tensors that have more than 2 dimensions. Found dimension = ')

    r, i, j, k = init_func(
        r_weight.size(1),
        r_weight.size(0),
        rng=rng,
        kernel_size=kernel_size,
        criterion=init_criterion
    )
    r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
    r_weight.data = r.type_as(r_weight.data)
    i_weight.data = i.type_as(i_weight.data)
    j_weight.data = j.type_as(j_weight.data)
    k_weight.data = k.type_as(k_weight.data)


def random_init(in_features, out_features, rng, kernel_size=None, criterion='glorot'):
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

    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    number_of_weights = np.prod(kernel_shape)
    v_r = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_i = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_j = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_k = np.random.uniform(-1.0, 1.0, number_of_weights)

    v_r = v_r.reshape(kernel_shape)
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    weight_r = v_r
    weight_i = v_i
    weight_j = v_j
    weight_k = v_k
    return (weight_r, weight_i, weight_j, weight_k)


def get_kernel_and_weight_shape(operation, in_channels, out_channels, kernel_size):
    if operation == 'convolution1d':
        if type(kernel_size) is not int:
            raise ValueError(
                """An invalid kernel_size was supplied for a 1d convolution. The kernel size
                must be integer in the case. Found kernel_size = """ + str(kernel_size)
            )
        else:
            ks = kernel_size
            w_shape = (out_channels, in_channels) + tuple((ks,))
    else:  # in case it is 2d or 3d.
        if operation == 'convolution2d' and type(kernel_size) is int:
            ks = (kernel_size, kernel_size)
        elif operation == 'convolution3d' and type(kernel_size) is int:
            ks = (kernel_size, kernel_size, kernel_size)
        elif type(kernel_size) is not int:
            if operation == 'convolution2d' and len(kernel_size) != 2:
                raise ValueError(
                    """An invalid kernel_size was supplied for a 2d convolution. The kernel size
                    must be either an integer or a tuple of 2. Found kernel_size = """ + str(kernel_size)
                )
            elif operation == 'convolution3d' and len(kernel_size) != 3:
                raise ValueError(
                    """An invalid kernel_size was supplied for a 3d convolution. The kernel size
                    must be either an integer or a tuple of 3. Found kernel_size = """ + str(kernel_size)
                )
            else:
                ks = kernel_size
        w_shape = (out_channels, in_channels) + (*ks,)
    return ks, w_shape


def quaternion_conv(input, r_weight, i_weight, j_weight, k_weight, bias, stride,
                    padding, groups, dilatation):
    """
    Applies a quaternion convolution to the incoming data:
    """

    cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=1)
    cat_kernels_4_i = torch.cat([i_weight, r_weight, -k_weight, j_weight], dim=1)
    cat_kernels_4_j = torch.cat([j_weight, k_weight, r_weight, -i_weight], dim=1)
    cat_kernels_4_k = torch.cat([k_weight, -j_weight, i_weight, r_weight], dim=1)

    cat_kernels_4_quaternion = torch.cat([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=0)

    if input.dim() == 3:
        convfunc = F.conv1d
    elif input.dim() == 4:
        convfunc = F.conv2d
    elif input.dim() == 5:
        convfunc = F.conv3d
    else:
        raise Exception("The convolutional input is either 3, 4 or 5 dimensions."
                        " input.dim = " + str(input.dim()))

    return convfunc(input, cat_kernels_4_quaternion, bias, stride, padding, dilatation, groups)


def quaternion_transpose_conv(input, r_weight, i_weight, j_weight, k_weight, bias, stride,
                              padding, output_padding, groups, dilatation):
    """
    Applies a quaternion trasposed convolution to the incoming data:

    """

    cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=1)
    cat_kernels_4_i = torch.cat([i_weight, r_weight, -k_weight, j_weight], dim=1)
    cat_kernels_4_j = torch.cat([j_weight, k_weight, r_weight, -i_weight], dim=1)
    cat_kernels_4_k = torch.cat([k_weight, -j_weight, i_weight, r_weight], dim=1)
    cat_kernels_4_quaternion = torch.cat([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=0)

    if input.dim() == 3:
        convfunc = F.conv_transpose1d
    elif input.dim() == 4:
        convfunc = F.conv_transpose2d
    elif input.dim() == 5:
        convfunc = F.conv_transpose3d
    else:
        raise Exception("The convolutional input is either 3, 4 or 5 dimensions."
                        " input.dim = " + str(input.dim()))

    return convfunc(input, cat_kernels_4_quaternion, bias, stride, padding, output_padding, groups, dilatation)


def quaternion_conv_rotation(input, zero_kernel, r_weight, i_weight, j_weight, k_weight, bias, stride,
                             padding, groups, dilatation, quaternion_format, scale=None):
    """
    Applies a quaternion rotation and convolution transformation to the incoming data:

    The rotation W*x*W^t can be replaced by R*x following:
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    Works for unitary and non unitary weights.

    The initial size of the input must be a multiple of 3 if quaternion_format = False and
    4 if quaternion_format = True.
    """

    square_r = (r_weight * r_weight)
    square_i = (i_weight * i_weight)
    square_j = (j_weight * j_weight)
    square_k = (k_weight * k_weight)

    norm = torch.sqrt(square_r + square_i + square_j + square_k + 0.0001)

    # print(norm)

    r_n_weight = (r_weight / norm)
    i_n_weight = (i_weight / norm)
    j_n_weight = (j_weight / norm)
    k_n_weight = (k_weight / norm)

    norm_factor = 2.0

    square_i = norm_factor * (i_n_weight * i_n_weight)
    square_j = norm_factor * (j_n_weight * j_n_weight)
    square_k = norm_factor * (k_n_weight * k_n_weight)

    ri = (norm_factor * r_n_weight * i_n_weight)
    rj = (norm_factor * r_n_weight * j_n_weight)
    rk = (norm_factor * r_n_weight * k_n_weight)

    ij = (norm_factor * i_n_weight * j_n_weight)
    ik = (norm_factor * i_n_weight * k_n_weight)

    jk = (norm_factor * j_n_weight * k_n_weight)

    if quaternion_format:
        if scale is not None:
            rot_kernel_1 = torch.cat(
                [zero_kernel, scale * (1.0 - (square_j + square_k)), scale * (ij - rk), scale * (ik + rj)], dim=1)
            rot_kernel_2 = torch.cat(
                [zero_kernel, scale * (ij + rk), scale * (1.0 - (square_i + square_k)), scale * (jk - ri)], dim=1)
            rot_kernel_3 = torch.cat(
                [zero_kernel, scale * (ik - rj), scale * (jk + ri), scale * (1.0 - (square_i + square_j))], dim=1)
        else:
            rot_kernel_1 = torch.cat([zero_kernel, (1.0 - (square_j + square_k)), (ij - rk), (ik + rj)], dim=1)
            rot_kernel_2 = torch.cat([zero_kernel, (ij + rk), (1.0 - (square_i + square_k)), (jk - ri)], dim=1)
            rot_kernel_3 = torch.cat([zero_kernel, (ik - rj), (jk + ri), (1.0 - (square_i + square_j))], dim=1)

        zero_kernel2 = torch.cat([zero_kernel, zero_kernel, zero_kernel, zero_kernel], dim=1)
        global_rot_kernel = torch.cat([zero_kernel2, rot_kernel_1, rot_kernel_2, rot_kernel_3], dim=0)

    else:
        if scale is not None:
            rot_kernel_1 = torch.cat([scale * (1.0 - (square_j + square_k)), scale * (ij - rk), scale * (ik + rj)],
                                     dim=0)
            rot_kernel_2 = torch.cat([scale * (ij + rk), scale * (1.0 - (square_i + square_k)), scale * (jk - ri)],
                                     dim=0)
            rot_kernel_3 = torch.cat([scale * (ik - rj), scale * (jk + ri), scale * (1.0 - (square_i + square_j))],
                                     dim=0)
        else:
            rot_kernel_1 = torch.cat([1.0 - (square_j + square_k), (ij - rk), (ik + rj)], dim=0)
            rot_kernel_2 = torch.cat([(ij + rk), 1.0 - (square_i + square_k), (jk - ri)], dim=0)
            rot_kernel_3 = torch.cat([(ik - rj), (jk + ri), (1.0 - (square_i + square_j))], dim=0)

        global_rot_kernel = torch.cat([rot_kernel_1, rot_kernel_2, rot_kernel_3], dim=0)

    # print(input.shape)
    # print(square_r.shape)
    # print(global_rot_kernel.shape)

    if input.dim() == 3:
        convfunc = F.conv1d
    elif input.dim() == 4:
        convfunc = F.conv2d
    elif input.dim() == 5:
        convfunc = F.conv3d
    else:
        raise Exception("The convolutional input is either 3, 4 or 5 dimensions."
                        " input.dim = " + str(input.dim()))

    return convfunc(input, global_rot_kernel, bias, stride, padding, dilatation, groups)


# Quaternion Block for Feature Contextualizer
class MAQ(nn.Module):
    def __init__(self, dim):
        super(MAQ, self).__init__()
        self.branch1 = CrossAttentionTransformer(dim)  # ACT
        self.branch2 = CrossAttentionTransformer(dim)   # KFT
        self.branch3 = SelfAttentionTransformer(dim)   # SAT
        self.qcnn = nn.Sequential(
            QuaternionConv(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.Conv2d(dim * 4, dim, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x, prior):
        x1 = self.branch1(prior, x)  # ACT
        x2 = self.branch2(x, prior)  # KFT
        x3 = self.branch3(x)  # SAT

        z = torch.zeros_like(x, device=x.device)
        out = torch.cat((z, x1, x2, x3), 1)
        out = self.qcnn(out)
        out = self.final(out)
        return out


class QBlock(nn.Module):
    def __init__(self, dim):
        super(QBlock, self).__init__()
        self.branch1 = RCB(dim)
        self.branch2 = NAFBlock(dim)

        self.qcnn = nn.Sequential(
            QuaternionConv(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.Conv2d(dim * 4, dim, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = torch.zeros_like(x, device=x.device)
        z = torch.zeros_like(x, device=x.device)

        out = torch.cat((z, x1, x2, x3), 1)
        out = self.qcnn(out)
        out = self.final(out)
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, groups=1, norm='in',
                 nonlinear='relu'):
        super(ConvLayer, self).__init__()
        reflection_padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups, bias=bias,
                                dilation=dilation)
        self.norm = norm
        self.nonlinear = nonlinear

        if norm == 'bn':
            self.normalization = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.normalization = nn.InstanceNorm2d(out_channels, affine=False)
        elif norm == 'ln':
            self.normalization = LayerNorm(dim=out_channels, LayerNorm_type='BiasFree')
        else:
            self.normalization = None

        if nonlinear == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif nonlinear == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif nonlinear == 'PReLU':
            self.activation = nn.PReLU()
        elif nonlinear == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = None

    def forward(self, x):
        out = self.conv2d(self.reflection_pad(x))

        if self.normalization is not None:
            out = self.normalization(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class SelfAttention(nn.Module):
    def __init__(self, channels, k, nonlinear='relu'):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.k = k
        self.nonlinear = nonlinear

        self.linear1 = nn.Linear(channels, channels // k)
        self.linear2 = nn.Linear(channels // k, channels)
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))

        if nonlinear == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif nonlinear == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif nonlinear == 'PReLU':
            self.activation = nn.PReLU()
        else:
            raise ValueError

    def attention(self, x):
        N, C, H, W = x.size()
        out = torch.flatten(self.global_pooling(x), 1)
        out = self.activation(self.linear1(out))
        out = torch.sigmoid(self.linear2(out)).view(N, C, 1, 1)

        return out.mul(x)

    def forward(self, x):
        return self.attention(x)


class Aggreation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, k=8, nonlinear='PReLU', norm='in'):
        super(Aggreation, self).__init__()
        self.attention = SelfAttention(in_channels, k, nonlinear='relu')
        self.conv = ConvLayer(in_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=1,
                              nonlinear=nonlinear,
                              norm=norm)

    def forward(self, x):
        res = self.conv(self.attention(x))
        return res


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c, embed_dim, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=4, interpolation_type='bilinear'):
        super(SPP, self).__init__()
        self.conv = nn.ModuleList()
        self.num_layers = num_layers
        self.interpolation_type = interpolation_type

        for _ in range(self.num_layers):
            self.conv.append(
                ConvLayer(in_channels, in_channels, kernel_size=1, stride=1, dilation=1, nonlinear='PReLU',
                          norm=None))

        self.fusion = ConvLayer((in_channels * (self.num_layers + 1)), out_channels, kernel_size=3, stride=1,
                                norm='False', nonlinear='PReLU')

    def forward(self, x):

        N, C, H, W = x.size()
        out = []

        for level in range(self.num_layers):
            out.append(F.interpolate(self.conv[level](
                F.avg_pool2d(x, kernel_size=2 * 2 ** (level + 1), stride=2 * 2 ** (level + 1),
                             padding=2 * 2 ** (level + 1) % 2)), size=(H, W), mode=self.interpolation_type))

        out.append(x)

        return self.fusion(torch.cat(out, dim=1))


class ColorBalancePrior(nn.Module):
    def __init__(self, ch_in=3):
        super(ColorBalancePrior, self).__init__()
        self.enc = NAFBlock(ch_in)

    def forward(self, x):
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_mean = x_mean.expand_as(x)

        prior = self.enc(x_mean)

        return prior

class PriorGuidedRE(nn.Module):
    def __init__(self, ch_in=3, down_depth=2):
        super(PriorGuidedRE, self).__init__()
        self.ch_in = ch_in
        self.down_depth = down_depth

        self.dr = nn.ModuleList()  # DetailRestorer
        self.downs = nn.ModuleList()
        self.prior_downs = nn.ModuleList()
        self.up = nn.ModuleList()
        self.fusion = nn.ModuleList()
        self.fc = FeatureContextualizer(ch_in=self.ch_in * 2 ** self.down_depth,
                                 ch_out=self.ch_in * 2 ** (self.down_depth + 1),
                                 dim=48)
        self.final = ScaleHarmonizer(self.ch_in * 3, self.ch_in)
        self.norm = nn.Sigmoid()

        for i in range(self.down_depth):
            self.downs.append(nn.Conv2d(self.ch_in * 2 ** i,
                                                self.ch_in * 2 ** (i + 1), kernel_size=2, stride=2))
            self.prior_downs.append(nn.Conv2d(self.ch_in * 2 ** i,
                                              self.ch_in * 2 ** (i + 1), kernel_size=2, stride=2))
            self.up.append(nn.Sequential(
                nn.Conv2d(self.ch_in * 2 ** (i + 1), self.ch_in * 2 ** i * 2 ** 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))

        for i in range(self.down_depth + 1):
            self.dr.append(DetailRestorer(self.ch_in * 2 ** i))
            self.fusion.append(ScaleHarmonizer(self.ch_in * 2 ** (i + 1), self.ch_in * 2 ** i))


    def forward(self, x, prior):
        dr_res = []
        dr_res.append(self.dr[0](x))
        prior_low = prior
        for i, (down, prior_down) in enumerate(zip(self.downs, self.prior_downs)):
            low = down(dr_res[i])
            dr_res.append(self.dr[i + 1](low))
            prior_low = prior_down(prior_low)

        fusion_res = []

        fusion_res.append(self.fusion[-1](self.fc(dr_res[-1], prior_low)))
        fusion_res[-1] = fusion_res[-1] + dr_res[-1]

        for i in reversed(range(0, self.down_depth)):
            fusion_res[-1] = self.up[i](fusion_res[-1])
            fusion_in = torch.cat((fusion_res[-1], dr_res[i]), dim=1)
            fusion_res.append(self.fusion[i](fusion_in))

        out = torch.cat((x, fusion_res[-1], prior), dim=1)
        out = self.norm(self.final(out))
        return out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.ch_in = 3
        self.down_depth = 2

        self.prior = ColorBalancePrior(self.ch_in)
        self.re = PriorGuidedRE(self.ch_in, self.down_depth)

    def forward(self, x):
        illum_prior = self.prior(x)

        out = self.re(x, illum_prior)

        return out



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model():
    model = Model().to(device)
    inp = torch.randn(1, 3, 256, 256).to(device)
    print("Model Summary:")
    summary(model, input_size=(1, 3, 256, 256))

    # Forward pass to print shapes
    print("\nForward Pass Shapes:")
    illum_prior = model.prior(inp)
    print(f"ColorBalancePrior output shape: {illum_prior.shape}")

    re_out = model.re(inp, illum_prior)
    print(f"PriorGuidedRE output shape: {re_out.shape}")

    final_out = model(inp)
    print(f"Model final output shape: {final_out.shape}")

if __name__ == '__main__':
    from thop import profile, clever_format
    inp = torch.randn(1, 3, 256, 256).to(device)
    model = Model().to(device)
    macs, params = profile(model, inputs=(inp,))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"MACs: {macs}, Params: {params}")
    test_model()

def test_shape_and_range(model):
    """
    Test that:
      - The output shape matches the input shape in (H,W).
      - The output range is within [0,1] if using Sigmoid.
    """
    print("Test: Shape & Range")
    model.eval()
    resolutions = [128, 256, 512]
    for res in resolutions:
        x = torch.randn(1, 3, res, res, device=device)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape, f"Mismatch: input {x.shape}, output {y.shape}"
        assert torch.all(y >= 0) and torch.all(y <= 1), f"Output values are outside [0,1], got min={y.min().item():.3f} max={y.max().item():.3f}"
    print("✓ test_shape_and_range passed.")

def test_model_summary(model):
    """
    Prints a summary of model layers & parameter counts.
    """
    print("Test: Model Summary")
    model.eval()
    info = summary(model, input_size=(1, 3, 256, 256), verbose=0)
    print(info)
    print("✓ test_model_summary done.")

def run_all_tests():
    """
    Runs each test in succession.
    """
    print("=== Running Tests on Model ===\n")
    model = Model().to(device)

    # Test 1: Basic shape & range
    test_shape_and_range(model)

    # Test 2: Model summary
    test_model_summary(model)

    print("\nAll tests completed!\n")

if __name__ == "__main__":
    run_all_tests()
