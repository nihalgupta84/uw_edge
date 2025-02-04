#File:  models/wavelet_model.py (our proposed model)
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
from numpy.random import RandomState
from scipy.stats import chi
from torchinfo import summary
from pytorch_wavelets import DWTForward
import torch.fft


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

class QuaternionChannelAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        # Edge detection kernel (combined X/Y Sobel)
        self.edge_conv = nn.Conv2d(3, 1, 3, padding=1, bias=False)
        sobel_kernel = torch.tensor([
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
             [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]],
             [[ 0, 0, 0], [ 0, 0, 0], [ 0, 0, 0]]]
        ], dtype=torch.float32).sum(dim=1, keepdim=True)
        self.edge_conv.weight.data = sobel_kernel.repeat(1,3,1,1)/8.0
        
        # Learnable refinement
        self.learnable_layer = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 1, 3, padding=1)
        )

    def forward(self, x):
        """
        x: (B,3,H,W) RGB input
        returns: (B,4,H,W) quaternion-ready tensor
        """
        # 1. Extract edge information
        edge_mag = torch.sqrt(torch.sum(self.edge_conv(x)**2, dim=1, keepdim=True))
        
        # 2. Learnable refinement
        edge_refined = self.learnable_layer(edge_mag)
        
        # 3. Adaptive scaling
        alpha = torch.sigmoid(edge_refined)  # [0,1] scaling
        edge_channel = alpha * x.mean(dim=1, keepdim=True) + (1-alpha) * edge_refined
        
        # 4. Concatenate with RGB
        return torch.cat([x, edge_channel], dim=1)

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        # print(f"LayerNormFunction: forward: x.shape={x.shape}, y.shape={y.shape}")
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
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
        self.channels = channels  # Store channels for validation

    def forward(self, x):
        if x.size(1) != self.channels:
            raise ValueError(f"Expected {self.channels} channels but got {x.size(1)}")
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


#########################################
# 1) MultiStageColorPrior
#########################################
class ColorUNet(nn.Module):
    """
    A U-Net that outputs a color offset map (B,3,H,W).
    We'll integrate it into MultiStageColorPrior for local color correction.
    """
    def __init__(self, in_ch=3, base_ch=16):
        super(ColorUNet, self).__init__()
        # -- encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        # -- bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch*4, base_ch*4, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # -- decoder
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # output color offset
        self.out_conv = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)               # (B,base_ch,H,W)
        p1 = self.pool1(e1)             # (B,base_ch,H/2,W/2)
        e2 = self.enc2(p1)              # (B,base_ch*2,H/2,W/2)
        p2 = self.pool2(e2)             # (B,base_ch*2,H/4,W/4)

        b  = self.bottleneck(p2)        # (B,base_ch*4,H/4,W/4)

        u2 = self.up2(b)                # (B,base_ch*2,H/2,W/2)
        cat2 = torch.cat([u2,e2], dim=1)# (B,base_ch*4,H/2,W/2)
        d2 = self.dec2(cat2)            # (B,base_ch*2,H/2,W/2)

        u1 = self.up1(d2)               # (B,base_ch,H,W)
        cat1 = torch.cat([u1,e1], dim=1)# (B,base_ch*2,H,W)
        d1 = self.dec1(cat1)            # (B,base_ch,H,W)

        color_offset = self.out_conv(d1)# (B,3,H,W)
        return color_offset


class MultiStageColorPrior(nn.Module):
    """
    1) Global color shift from average color.
    2) U-Net-based local color offset map.
    """
    def __init__(self, in_ch=3, hidden_dim=16, unet_ch=16):
        super(MultiStageColorPrior, self).__init__()
        self.in_ch = in_ch

        # (A) Global shift
        self.global_linear = nn.Sequential(
            nn.Linear(in_ch, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_ch)
        )

        # (B) U-Net for local color
        self.color_unet = ColorUNet(in_ch=in_ch, base_ch=unet_ch)

    def forward(self, x):
        B, C, H, W = x.shape

        # Step A: Check x itself
        print("ColorPrior Step A: x mean/std =", x.mean().item(), x.std().item())

        # Step B: global shift
        avg_color = torch.mean(x.view(B, C, -1), dim=2)
        print("ColorPrior Step B: avg_color =", avg_color)  # shape=(B,3)
        shift = self.global_linear(avg_color).view(B,C,1,1)
        print("ColorPrior Step B: shift mean/std =", shift.mean().item(), shift.std().item())

        global_corrected = x + shift
        print("ColorPrior Step B: global_corrected mean/std =", 
            global_corrected.mean().item(), global_corrected.std().item())

        # Step C: color_unet
        color_offset = self.color_unet(global_corrected)
        print("ColorPrior Step C: color_offset mean/std =", 
            color_offset.mean().item(), color_offset.std().item())

        color_prior  = global_corrected + color_offset
        print("ColorPrior Step C: color_prior mean/std =", 
            color_prior.mean().item(), color_prior.std().item())
        return color_prior

def clamp_subband(x, min_val=1e-6, max_val=1e+6):
    """
    Clamps absolute values to avoid meltdown from too-small or too-large magnitudes.
    Ensures no value is below min_val in magnitude, or above max_val.
    """
    # If you want to zero out extremely small values:
    # x = torch.where(x.abs() < min_val, torch.zeros_like(x), x)
    # Alternatively, just clamp to ±min_val:
    x = torch.clamp(x, min=-max_val, max=max_val)
    too_small = (x.abs() < min_val)
    if too_small.any():
        x = torch.where(too_small, x.new_zeros([]), x)
    return x

class SubBandAttention(nn.Module):
    """
    Simple scalar gating for LH, HL, HH.
    """
    def __init__(self, num_bands=3):
        super(SubBandAttention, self).__init__()
        self.attn_params = nn.Parameter(0.1 * torch.ones(num_bands), requires_grad=True)
    
    def forward(self, lh, hl, hh):
        raw = self.attn_params  # shape (3,)
        attn = 0.01 + 0.98 * torch.sigmoid(raw)
        print(f"subband attention: raw={raw}, attn={attn}")

        lh_out = lh * attn[0]
        lh_out = clamp_subband(lh_out)  # <-- newly added clamp
        print(f" lh_out shape , mean and standard deviation in subband attention: {lh_out.shape}, {lh_out.mean()}, {lh_out.std()}")

        hl_out = hl * attn[1]
        hl_out = clamp_subband(hl_out)  # <-- newly added clamp
        print(f" hl_out shape , mean and standard deviation in subband attention: {hl_out.shape}, {hl_out.mean()}, {hl_out.std()}")

        hh_out = hh * attn[2]
        hh_out = clamp_subband(hh_out)  # <-- newly added clamp
        print(f" hh_out shape , mean and standard deviation in subband attention: {hh_out.shape}, {hh_out.mean()}, {hh_out.std()}")

        return lh_out, hl_out, hh_out

class LowResWaveletPath(nn.Module):
    """
    Processes sub-bands at half resolution, then the caller up-samples.
    """
    def __init__(self, in_ch=3, mid_ch=16):
        super(LowResWaveletPath, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1)
        self.act1  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_ch, in_ch, 3, padding=1)
        self.norm = nn.InstanceNorm2d(in_ch)
    
    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.conv2(x)
        x = safe_instancenorm(x)
        print(f" LowResWaveletPath shape , mean and standard deviation : {x.shape}, {x.mean()}, {x.std()}") 
        return x  
def safe_instancenorm(x, eps=1e-8):
    B,C,H,W = x.shape
    mu = x.mean(dim=[2,3], keepdim=True)
    var = x.var(dim=[2,3], keepdim=True, unbiased=False)
    # clamp variance
    var = torch.clamp(var, min=1e-12)
    x_norm = (x - mu) / torch.sqrt(var + eps)
    return x_norm
    
class FFTBranch(nn.Module):
    """Enhanced FFT branch with guaranteed activation and proper processing"""
    def __init__(self, in_ch=3, mid_ch=16, out_ch=6):
        super(FFTBranch, self).__init__()
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch

        # Magnitude processing
        self.mag_net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, in_ch, 3, padding=1)
        )
        
        # Phase processing
        self.phase_net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, in_ch, 3, padding=1)
        )
        
        # Final fusion layer
        self.fusion = nn.Conv2d(in_ch * 2, out_ch, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, mag, phase):
        """Process magnitude and phase with guaranteed activation"""
        # Ensure proper device placement
        if not self.training:  # Cache during inference
            self._mag_cache = self.mag_net(mag)
            self._phase_cache = self.phase_net(phase)
        else:
            self._mag_cache = None
            self._phase_cache = None
            
        # Process components
        mag_feat = self.mag_net(mag)
        
        phase_feat = self.phase_net(phase)
        print(f" mag shape , mean and standard deviation in FFT Branch : {mag_feat.shape}, {mag_feat.mean()}, {mag_feat.std()}")
        print(f" phase shape , mean and standard deviation in FFT Branch: {phase_feat.shape}, {phase_feat.mean()}, {phase_feat.std()}")
        # Concatenate and fuse
        combined = torch.cat([mag_feat, phase_feat], dim=1)
        out = self.fusion(combined)
        
        return out
    
    def get_last_activations(self):
        """Return cached activations (for testing/monitoring)"""
        return {
            'magnitude': self._mag_cache,
            'phase': self._phase_cache
        } if not self.training else None

#########################################
# 2) Wavelet + FFT Decomposition
#########################################

class WaveletFFTDecomposition(nn.Module):
    def __init__(self, wavelet='haar'):
        super(WaveletFFTDecomposition, self).__init__()
        # Single-level DWT with pytorch_wavelets
        self.dwt = DWTForward(J=1, wave=wavelet)

    def forward(self, x):
        """
        x: (B,3,H,W)
        Returns: LL, LH, HL, HH, mag, phase
        """
        # DWT -> (LL, [LH,HL,HH])
        Yl, Yh = self.dwt(x)  # Yl: (B, C, H//2, W//2), Yh[0]: (B,C,3,H//2,W//2)
        LL = Yl
        LH = Yh[0][:, :, 0]
        HL = Yh[0][:, :, 1]
        HH = Yh[0][:, :, 2]

        # FFT
        fft_complex = torch.fft.fft2(x, dim=(-2, -1))
        print(f" fft_complex shape , mean and standard deviation in wavelet decompostion class: {fft_complex.shape}, {fft_complex.mean()}, {fft_complex.std()}")
        mag = torch.abs(fft_complex)
        phase = torch.angle(fft_complex)
        print(f" mag shape , mean and standard deviation in waveletdecompostionclass : {mag.shape}, {mag.mean()}, {mag.std()}")
        print(f" phase shape , mean and standard deviation in waveletdecomostion class : {phase.shape}, {phase.mean()}, {phase.std()}")
        return LL, LH, HL, HH, mag, phase
    
#########################################
# 3) Cross-Attention Modules
#########################################

class FeedForward(nn.Module):
    """
    Standard feed-forward block used in transformers.
    """
    def __init__(self, dim_in, expansion_factor=2.66, bias=True):
        super(FeedForward, self).__init__()
        hidden_dim = int(dim_in * expansion_factor)
        self.conv1 = nn.Conv2d(dim_in, hidden_dim*2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_dim*2, hidden_dim*2, 3, padding=1, groups=hidden_dim*2, bias=bias)
        self.conv2 = nn.Conv2d(hidden_dim, dim_in, 1, bias=bias)
    
    def forward(self, x):
        # (B, C, H, W)
        x_in = x
        x = self.conv1(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.conv2(x)
        return x + x_in


class CrossAttention(nn.Module):
    """
    Cross-attention that merges two input feature maps Q and K,
    typical of 'query' vs 'key/value' in transformers.
    """
    def __init__(self, dim, num_heads=2, bias=True):
        super(CrossAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.q  = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
    def forward(self, x_q, x_k):
        """
        x_q: query  (B, C, H, W)
        x_k: key    (B, C, H, W)
        """
        B, C, H, W = x_q.shape
        
        # Compute Q,K,V from x_q, x_k
        q = self.q(x_q)        # (B, C, H, W)
        kv = self.kv(x_k)      # => (B, 2C, H, W)
        k, v = torch.chunk(kv, 2, dim=1)
        
        # Reshape to (B, heads, C//heads, H*W)
        def reshape_for_attn(tensor):
            return tensor.view(B, self.num_heads, C//self.num_heads, -1)
        
        q = reshape_for_attn(q)
        k = reshape_for_attn(k)
        v = reshape_for_attn(v)
        
        # L2 normalize
        q = F.normalize(q, dim=2)
        k = F.normalize(k, dim=2)
        print(f"cross_attention q shape , mean and standard deviation in cross attention class: {q.shape}, {q.mean()}, {q.std()}")
        print(f"cross_attention k shape , mean and standard deviation in cross attention class : {k.shape}, {k.mean()}, {k.std()}")
        print(f"cross_attention v shape , mean and standard deviation in cross attention class: {v.shape}, {v.mean()}, {v.std()}")
        # dot product
        attn_scale = 0.1 * torch.sigmoid(self.temperature)  # range ~ 0..0.1
        attn = torch.einsum('bhcd,bhkd->bhck', q, k) * attn_scale
        attn = F.softmax(attn, dim=-1)  # shape => (B, heads, C//heads, C//heads)
        print(f"cross_attention attn shape , mean and standard deviation in cross attention class: {attn.shape}, {attn.mean()}, {attn.std()}")
        # Weighted sum
        out = torch.einsum('bhck,bhkd->bhcd', attn, v)
        print(f"cross_attention output shape , mean and standard deviation in cross attention class: {out.shape}, {out.mean()}, {out.std()}")
        # reshape back
        out = out.contiguous().view(B, C, H, W)
        
        out = self.proj_out(out)
        print(f"cross_attention output shape , mean and standard deviation in cross attention class: {out.shape}, {out.mean()}, {out.std()}")
        return out


class CrossAttentionBlock(nn.Module):
    """
    One block of cross-attn + feed-forward, reminiscent of transformer encoder.
    """
    def __init__(self, dim, num_heads=2, ffn_factor=2.66, bias=True):
        super(CrossAttentionBlock, self).__init__()

        self.norm1 = LayerNorm2d(dim)  # 2D layernorm
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, bias=bias)
        
        self.norm2 = LayerNorm2d(dim)
        self.ffn = FeedForward(dim, ffn_factor, bias=bias)
    
    def forward(self, x, y):
        # "y" is the only updated variable
        y_in = y
        x_ln = self.norm1(x)
        y_ln = self.norm1(y)
        y_attn = self.cross_attn(x_ln, y_ln)
        y = y_in + y_attn
        y = y + self.ffn(self.norm2(y))
        print("cross_block1 input x mean/std in in cross attention block class:", x.shape, x.mean().item(), x.std().item())
        print("cross_block1 input y mean/std in cross attention block class:", y.shape, y.mean().item(), y.std().item())           
        return y



#########################################
# 4) Quaternion-based Feature Extractor
#########################################
# (Reusing your QuaternionConv code can be done here; for brevity, we do a short version.)

class SimpleQuaternionFeatureExtractor(nn.Module):
    """
    A small stack of quaternion convolutions + activations to capture cross-channel interactions.
    For demonstration, we do 2 quaternion conv layers.
    """
    def __init__(self, in_ch=3, out_ch=16):
        super(SimpleQuaternionFeatureExtractor, self).__init__()
        
        assert out_ch % 4 == 0, "out_ch must be multiple of 4"
        # Channel adapter with normalization
        self.channel_adapter = QuaternionChannelAdapter()
        self.input_norm = LayerNorm2d(4)  # Normalize after adaptation
        
        # First quaternion block
        self.qconv1 = QuaternionConv(
            in_channels=4,
            out_channels=out_ch*4,
            kernel_size=3,
            stride=1,
            padding=1,
            rotation=False
        )
        self.norm1 = LayerNorm2d(out_ch*4)
        self.act1 = nn.SiLU(inplace=True)
        
        # Second quaternion block
        self.qconv2 = QuaternionConv(
            in_channels=out_ch*4,
            out_channels=out_ch*4,
            kernel_size=3,
            stride=1,
            padding=1,
            rotation=False
        )
        self.norm2 = LayerNorm2d(out_ch*4)
        self.act2 = nn.SiLU(inplace=True)
        
        # Final reduction
        self.final = nn.Sequential(
            nn.Conv2d(out_ch*4, out_ch, 1),
            LayerNorm2d(out_ch)
        )
        
        # Activation tracking for debugging
        self._last_activations = {}
    
    def forward(self, x):
        # Track activations
        def save_activation(name):
            def hook(module, input, output):
                self._last_activations[name] = output.detach()
            return hook
        
        # Register hooks if not already registered
        if not hasattr(self, '_hooks'):
            self._hooks = [
                self.norm1.register_forward_hook(save_activation('norm1')),
                self.norm2.register_forward_hook(save_activation('norm2'))
            ]
        
        # Process input
        x_adapt = self.channel_adapter(x)
        print("Trunk input after channel_adapter shape/mean/std in simplequaternion extractor class:", x.shape,
          x_adapt.mean().item(), x_adapt.std().item())
        x = self.input_norm(x_adapt)
        
        # First block
        x = self.qconv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        # Second block
        x = self.qconv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        
        # Final reduction
        x = self.final(x)
        print(f"SimpleQuaternionFeatureExtractor: x shape/mean/std in simplequaternion extractor class: {x.shape}, {x.mean().item()}, {x.std().item()}")
        
        return x
    
    def get_activation_stats(self):
        """Return normalization layer statistics"""
        return {name: {
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'shape': tensor.shape
        } for name, tensor in self._last_activations.items()}
    

#########################################
# 5) Detail Restoration / Multi-scale
#########################################
class QBlock(nn.Module):
    """
    A small quaternion-based residual block:
      - 2 quaternion convs + skip
    """
    def __init__(self, ch):
        super(QBlock, self).__init__()
        self.qconv1 = QuaternionConv(
            in_channels=ch,
            out_channels=ch,
            kernel_size=3,
            stride=1,  # Critical fix
            padding=1,
            rotation=False
        )
        self.act1   = nn.ReLU(inplace=True)
        self.qconv2 = QuaternionConv(
            in_channels=ch,
            out_channels=ch,
            kernel_size=3,
            stride=1,  # Critical fix
            padding=1,
            rotation=False
        )
    
    def forward(self, x):
        res = x
        x = self.act1(self.qconv1(x))
        print("After qconv1 shape/mean/std in qblock clas:", x.shape, x.mean().item(), x.std().item())
        x = self.qconv2(x)
        print("After qconv2 shape/mean/std in qblock clas:", x.shape, x.mean().item(), x.std().item())
        x = F.layer_norm(x, x.shape[1:]) 
        print(f"x in qblock  shape/mean/std in qblock clas: {x.shape}, {x.mean().item()}, {x.std().item()}")
        return x + res


class DeeperDetailRestorer(nn.Module):
    """
    Stacks multiple QBlocks for richer detail enhancement.
    """
    def __init__(self, ch=16, num_blocks=3):
        super(DeeperDetailRestorer, self).__init__()
        self.blocks = nn.ModuleList([QBlock(ch) for _ in range(num_blocks)])
        self.final_act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final_act(x)
        print(f"DeeperDetailRestorer: x shape/mean/std in deepdetailrestore class: {x.shape}, {x.mean().item()}, {x.std().item()}")
        return x



class DetailRestoration(nn.Module):
    """
    A simple multi-scale detail restorer using standard residual blocks or NAFBlocks.
    You can expand this to have multiple repeated blocks for advanced detail.
    """
    def __init__(self, ch=16):
        super(DetailRestoration, self).__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act1  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
    
    def forward(self, x):
        """
        x: (B, ch, H, W)
        returns: detail-refined out
        """
        residual = x
        x = self.act1(self.conv1(x))
        x = self.conv2(x)
        x += residual
        print(f"DetailRestoration: x shape/mean/std in detail restore class: {x.shape}, {x.mean().item()}, {x.std().item()}")
        return x


#########################################
# 6) SPP for multi-scale context
#########################################
class SPP(nn.Module):
    """
    Spatial Pyramid Pooling: gather multi-scale pooling context.
    """
    def __init__(self, in_ch, out_ch, num_levels=3, interp='bilinear'):
        super(SPP, self).__init__()
        self.num_levels = num_levels
        self.interp = interp
        
        # We do small 1x1 conv for each pooled scale
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
            for _ in range(num_levels)
        ])
        
        self.fusion = nn.Conv2d(in_ch*(num_levels+1), out_ch, 3, padding=1, bias=False)
    
    def forward(self, x):
        B, C, H, W = x.shape
        out_list = []
        
        for i in range(self.num_levels):
            # e.g. pool_size = 2^(i+1)
            k = 2**(i+1)
            pooled = F.avg_pool2d(x, kernel_size=k, stride=k)  # shape => (B, C, H/k, W/k)
            feat = self.convs[i](pooled)
            up = F.interpolate(feat, size=(H, W), mode=self.interp)
            out_list.append(up)
        
        out_list.append(x)  # original
        out_cat = torch.cat(out_list, dim=1)  # => (B, C*(num_levels+1), H, W)
        print(f"SPP: out_cat shape/mean/std in spp class: {out_cat.shape}, {out_cat.mean().item()}, {out_cat.std().item()}")
        out = self.fusion(out_cat)
        print(f"SPP: out shape/mean/std in spp class: {out.shape}, {out.mean().item()}, {out.std().item()}")
        return out
class FinalScaleShift(nn.Module):
    """
    Learns a scale (alpha) and shift (beta) per channel.
    For an RGB image, it has shape (1,3,1,1).
    """
    def __init__(self, num_channels=3):
        super(FinalScaleShift, self).__init__()
        # Initialize scale to 1.0, shift to 0.0
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        # x: (B,3,H,W)
        # scale + shift for each channel
        print(f"FinalScaleShift: x shape/mean/std: {x.shape}, {x.mean().item()}, {x.std().item()}")
        return x * self.scale + self.shift


#########################################
# 7) Final Model: "FinalUnderwaterEnhancer"
#########################################
class WaveletModel_V1(nn.Module):
    def __init__(self,
                 base_ch=32,
                 wavelet='haar',
                 use_subband_attn=True,
                 use_fft_branch=True,
                 deeper_detail=True):
        super(WaveletModel_V1, self).__init__()

        assert base_ch % 4 == 0, "base_ch must be multiple of 4 for quaternion convs"
        self.base_ch = base_ch

        # (1) color prior
        self.color_prior = MultiStageColorPrior(in_ch=3, hidden_dim=16, unet_ch=16)

        # (2) wavelet + FFT decomposition (differentiable!)
        self.spectral_decomp = WaveletFFTDecomposition(wavelet=wavelet)
        self.use_subband_attn = use_subband_attn
        if use_subband_attn:
            self.subband_attn = SubBandAttention(num_bands=3)
            self.lowres_path = LowResWaveletPath(in_ch=3, mid_ch=16)

        self.use_fft_branch = use_fft_branch
        if use_fft_branch:
            self.fft_branch = FFTBranch(in_ch=3, mid_ch=16, out_ch=6)

        # (3) quaternion trunk
        self.qfeat = SimpleQuaternionFeatureExtractor(in_ch=3, out_ch=base_ch)

        # (4) cross-attn
        self.cross_block1 = CrossAttentionBlock(dim=base_ch, num_heads=2)
        self.cross_block2 = CrossAttentionBlock(dim=base_ch, num_heads=2)

        # (5) detail restoration
        if deeper_detail:
            self.detail_restorer = DeeperDetailRestorer(ch=base_ch, num_blocks=3)
        else:
            self.detail_restorer = DetailRestoration(ch=base_ch)

        # (6) SPP
        self.spp = SPP(in_ch=base_ch, out_ch=base_ch, num_levels=3)

        # (7) final step
        self.final_out = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)
        self.final_harmonizer = FinalScaleShift(num_channels=3)
        self.act = nn.Sigmoid()

        # unify dims
        self.color_reduce = nn.Conv2d(3, base_ch, kernel_size=1, bias=False)
        self.spec_reduce = nn.Conv2d(18, base_ch, kernel_size=1, bias=False)

        # init
        self.apply(init_weights)
    def forward(self, x):
        B, C, H, W = x.shape

        # 1) color prior => (B,3,H,W) 
        color_feat = self.color_prior(x)  # => (B,3,H,W)
        print("Color prior feature shape/mean/standard deviation in waveletmodel:", color_feat.shape, color_feat.mean().item(), color_feat.std().item())

        # 2) wavelet + FFT => (LL,LH,HL,HH,mag,phase)
        LL, LH, HL, HH, mag, phase = self.spectral_decomp(x)
        print("Wavelet decomposition shapes/mean/standard deviation in wavelet model: LL:", LL.shape, LL.mean().item(), LL.std().item(), "LH:", LH.shape, LH.mean().item(), LH.std().item(), "HL:", HL.shape, HL.mean().item(), HL.std().item(), "HH:", HH.shape, HH.mean().item(), HH.std().item())
        # LL.shape = > (B,3,H/2,W/2), LH.shape = > (B,3,H/2,W/2), HL.shape = > (B,3,H/2,W/2), HH.shape = > (B,3,H/2,W/2)           
        print("FFT decomposition shapes/mean/standard deviation in wavelet model: mag:", mag.shape, mag.mean().item(), mag.std().item(), "phase:", phase.shape, phase.mean().item(), phase.std().item())     
        # mag.shape = > (B,3,H,W), phase.shape = > (B,3,H,W)
        if self.use_subband_attn:
            LH, HL, HH = self.subband_attn(LH, HL, HH)
            print("Sub-band attention applied")
            LH_proc = self.lowres_path(LH)
            HL_proc = self.lowres_path(HL)
            HH_proc = self.lowres_path(HH)
            LH_up = F.interpolate(LH_proc, size=(H, W), mode='bilinear')
            HL_up = F.interpolate(HL_proc, size=(H, W), mode='bilinear')
            HH_up = F.interpolate(HH_proc, size=(H, W), mode='bilinear')
            LL_up = F.interpolate(LL, size=(H, W), mode='bilinear')
        else:
            LH_up = F.interpolate(LH, size=(H, W), mode='bilinear')
            HL_up = F.interpolate(HL, size=(H, W), mode='bilinear')
            HH_up = F.interpolate(HH, size=(H, W), mode='bilinear')
            LL_up = F.interpolate(LL, size=(H, W), mode='bilinear')
        print("Interpolated wavelet sub-bands shapes/mean/stanndard deviation in wavelet model: LL_up:", LL_up.shape, LL_up.mean().item(), LL_up.std().item(), "LH_up:", LH_up.shape, LH_up.mean().item(), LH_up.std().item(), "HL_up:", HL_up.shape, HL_up.mean().item(), HL_up.std().item(), "HH_up:", HH_up.shape, HH_up.mean().item(), HH_up.std().item())
        # LL_up.shape = > (B,3,H,W), LH_up.shape = > (B,3,H,W), HL_up.shape = > (B,3,H,W), HH_up.shape = > (B,3,H,W)
        print("LH_up mean/std:", LH_up.mean().item(), LH_up.std().item())
        print("HL_up mean/std:", HL_up.mean().item(), HL_up.std().item())
        print("HH_up mean/std:", HH_up.mean().item(), HH_up.std().item())
        # combine wavelet sub-bands + optional FFT
        if self.use_fft_branch:
            fft_out = self.fft_branch(mag, phase)  # => (B,6,H,W)
            print("FFT Branch output shape/mean/std in wavelet model:", fft_out.shape, fft_out.mean().item(), fft_out.std().item())
            # fft_out.shape = > (B,6,H,W)
            spectral_feat = torch.cat([LL_up, LH_up, HL_up, HH_up, fft_out], dim=1)  # =>(B,3+3+3+3+6,H,W)=>(B,18,H,W)
        else:
            spectral_feat = torch.cat([LL_up, LH_up, HL_up, HH_up, mag, phase], dim=1)  # =>(B,18,H,W)
        print("spectral_feat shape/mean/standard deviation in wavelet model", spectral_feat.shape, spectral_feat.mean().item(), spectral_feat.std().item())
        # spectral_feat.shape = > (B,18,H,W)

        # # 3) quaternion trunk => (B,base_ch,H,W)
        trunk_feat = self.qfeat(x)
        print("Quaternion trunk feature shape/mean/standard deviation in wavelet model:", trunk_feat.shape, trunk_feat.mean().item(), trunk_feat.std().item())
        # # trunk_feat.shape = > (B,64,H,W)


        # # 4a) cross_block1 merges trunk & color
        color_aligned = self.color_reduce(color_feat)  # =>(B,base_ch,H,W)
        print("Color aligned feature shape/mean/standard deviation in wavelet model:", color_aligned.shape, color_aligned.mean().item(), color_aligned.std().item())

        # # color_aligned.shape = > (B,64,H,W)


        y1 = self.cross_block1(color_aligned, trunk_feat)
        print("Cross-attention block 1 output shape/mean/standard deviation in wavelet model:", y1.shape, y1.mean().item(), y1.std().item())
        # # y1.shape = > (B,64,H,W)

        # # 4b) cross_block2 merges trunk & wavelet

        spectral_aligned = self.spec_reduce(spectral_feat)
        print("Spectral aligned feature shape/mean/standard deviation in wavelet model:", spectral_aligned.shape, spectral_aligned.mean().item(), spectral_aligned.std().item())
        # # # spectral_aligned.shape = > (B,64,H,W)

        y2 = self.cross_block2(y1, spectral_aligned)
        print("Cross-attention block 2 output shape/mean/standard deviation in wavelet model:", y2.shape, y2.mean().item(), y2.std().item())
        # # # y2.shape = > (B,64,H,W)

        # # 5) detail => (B,base_ch,H,W)
        detail_out = self.detail_restorer(y2)
        print("Detail restoration output shape/mean/standard deviation in wavelet model:", detail_out.shape, detail_out.mean().item(), detail_out.std().item())
        # # detail_out.shape = > (B,64,H,W)

        # # 6) spp => (B,base_ch,H,W)
        spp_out = self.spp(detail_out)
        print("SPP output shape/mean/standard deviation in wavelet model:", spp_out.shape, spp_out.mean().item(), spp_out.std().item())
        # # spp_out.shape = > (B,64,H,W)

        # # 7) final => (B,3,H,W)
        out = self.final_out(spp_out)
        print("Final output shape/mean/standard deviation in wavelet model:", out.shape, out.mean().item(), out.std().item())
        out = self.final_harmonizer(out)
        print("Final output shape/mean/standard deviation in wavelet model:", out.shape, out.mean().item(), out.std().item())
        out = self.act(out)
        print("Final output shape/mean/standard deviation in wavelet model:", out.shape, out.mean().item(), out.std().item())
        # === Newly added: final nan-check and clamp ===
        if torch.isnan(out).any():
            print("WARNING: Detected NaN in final output. Clamping to zero.")
            out = torch.nan_to_num(out, nan=0.0)  # or out.fill_(0.0)

        # Also ensure range is [0,1] if LPIPS requires that
        out = torch.clamp(out, 0.0, 1.0)
        return out


########################################################
# Keep your other modules: quaternion conv, color prior,
# cross attention blocks, SPP, final scale/shift, etc.
# with minimal or no changes, as above.
########################################################

def init_weights(m):
    """
    Custom initialization for your network. 
    Keep as before, but ensure you do NOT skip
    gradient flow anywhere by .detach() or .numpy().
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, LayerNorm2d)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


import torch
import torch.nn as nn
import torch.optim as optim

def check_gradient_flow(model, device='cuda'):
    torch.autograd.set_detect_anomaly(True)
    
    # Move model to device
    model = model.to(device)
    model.train()  # Must be in train mode to allow gradient updates

    # 1) Dummy input
    B, C, H, W = 2, 3, 256, 256
    dummy_input = torch.rand(B, C, H, W, device=device)
    dummy_input.requires_grad_()  # not strictly necessary, but clarifies debugging
    
    # 2) Forward pass
    output = model(dummy_input)

    # 3) Create a target that is definitely different from 'output'
    #    e.g. all ones
    target = torch.ones_like(output) * 0.5  # or 0.8, or some distinct value

    criterion = nn.MSELoss()
    loss = criterion(output, target)
    print("Loss:", loss.item())

    # 4) Backward pass
    model.zero_grad()
    loss.backward()

    # 5) Inspect gradient norms
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"Param: {name:<50} | NO GRAD")
        else:
            grad_norm = param.grad.data.norm(2)
            print(f"Param: {name:<50} | Grad Norm: {grad_norm:.6f}")
            # Check for NaN or Inf
            if torch.isnan(grad_norm):
                print(" -> NaN detected in grad!")
            if torch.isinf(grad_norm):
                print(" -> Inf detected in grad!")
        def fw_hook(m, inp, out):
            print(f"Parameter {name} used in forward!")
            param.register_hook(fw_hook)

# USAGE EXAMPLE
if __name__ == "__main__":
    # from wavelet_model import WaveletModel

    # Instantiate your model
    model = WaveletModel(base_ch=64, wavelet='haar')

    # Run our gradient flow check
    check_gradient_flow(model, device='cuda')
