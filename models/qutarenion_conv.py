#File: models/qutarenion_conv.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from numpy.random import RandomState
from scipy.stats import chi
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
