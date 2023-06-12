# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from __future__ import division
from typing import Tuple, Callable

import torch
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init

# Parts of this code are adapted from (https://github.com/timothygebhard/pytorch-conv4d)
# MIT License
# Copyright (c) 2019 Timothy Gebhard
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class Conv4d(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int, int, int], # temp, x,y,z
                 stride: Tuple[int, int, int, int] = [1,1,1,1], # temp, x,y,z
                 padding: Tuple[int, int, int, int] = [0,0,0,0], # temp, x,y,z
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 bias_initializer: Callable = None,
                 kernel_initializer: Callable = None):

        super(Conv4d, self).__init__()

        # ---------------------------------------------------------------------
        # Assertions for constructor arguments
        # ---------------------------------------------------------------------

        assert len(kernel_size) == 4, \
            '4D kernel size expected!'
        assert len(stride) == 4, \
            '4D Stride size expected!'
        assert dilation == 1, \
            'Dilation rate other than 1 not yet implemented!'
        assert groups == 1, \
            'Groups other than 1 not yet implemented!'

        # ---------------------------------------------------------------------
        # Store constructor arguments
        # ---------------------------------------------------------------------

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.stride = stride

        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer

        # ---------------------------------------------------------------------
        # Construct 3D convolutional layers
        # ---------------------------------------------------------------------

        # Shortcut for kernel dimensions
        (l_k, d_k, h_k, w_k) = self.kernel_size

        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv3d_layers = torch.nn.ModuleList()

        for i in range(l_k):

            # Initialize a Conv3D layer
            conv3d_layer = torch.nn.Conv3d(in_channels=self.in_channels,
                                           out_channels=self.out_channels,
                                           kernel_size=(d_k, h_k, w_k),
                                           stride=(self.stride[1], self.stride[2], self.stride[3]),
                                           padding=(self.padding[1],self.padding[2],self.padding[3]))

            # Apply initializer functions to weight and bias tensor
            if self.kernel_initializer is not None:
                self.kernel_initializer(conv3d_layer.weight)
            if self.bias_initializer is not None:
                self.bias_initializer(conv3d_layer.bias)

            # Store the layer
            self.conv3d_layers.append(conv3d_layer)

    # -------------------------------------------------------------------------

    def forward(self, input):

        # Define shortcut names for dimensions of input and kernel
        (b, c_i, l_i, d_i, h_i, w_i) = tuple(input.shape)
        (l_k, d_k, h_k, w_k) = self.kernel_size

        # Compute the size of the output tensor based on the zero padding
        (l_o, d_o, h_o, w_o) = (l_i + 2 * self.padding[0] - l_k + 1,
                                d_i + 2 * self.padding[1] - d_k + 1,
                                h_i + 2 * self.padding[2] - h_k + 1,
                                w_i + 2 * self.padding[3] - w_k + 1)

        # Output tensors for each 3D frame
        frame_results = l_o * [None]
        Stride_Temporär = self.stride[0]
        # Convolve each kernel frame i with each input frame j
        for i in range(l_k):
            Count = 0
            for j in range(l_i):

                # Add results to this output frame
                out_frame = j - (i - l_k // 2) - (l_i - l_o) // 2
                if out_frame < 0 or out_frame >= l_o:
                    continue
                if (Count % Stride_Temporär == 0):   #Apply Temporal Stride here
                    frame_conv3d = \
                        self.conv3d_layers[i](input[:, :, j, :]
                                            .view(b, c_i, d_i, h_i, w_i))

                    Count = Count +  1
                    if frame_results[out_frame] is None:
                        frame_results[out_frame] = frame_conv3d
                    else:
                        frame_results[out_frame] += frame_conv3d
                else:
                    Count = Count +  1 

        return torch.stack(frame_results[0::Stride_Temporär], dim=2)

class Pool4d(torch.nn.Module):
    def __init__(self,
                 kernel_size: Tuple[int, int, int, int], # temp, x,y,z
                 stride: Tuple[int, int, int, int], # temp, x,y,z
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 bias_initializer: Callable = None,
                 kernel_initializer: Callable = None):

        super(Pool4d, self).__init__()

        # ---------------------------------------------------------------------
        # Assertions for constructor arguments
        # ---------------------------------------------------------------------

        assert len(kernel_size) == 4, \
            '4D kernel size expected!'
        assert len(stride) == 4, \
            '4D Stride size expected!'
        assert dilation == 1, \
            'Dilation rate other than 1 not yet implemented!'
        assert groups == 1, \
            'Groups other than 1 not yet implemented!'

        # ---------------------------------------------------------------------
        # Store constructor arguments
        # ---------------------------------------------------------------------
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.stride = stride

        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer

        # ---------------------------------------------------------------------
        # Construct 3D convolutional layers
        # ---------------------------------------------------------------------

        # Shortcut for kernel dimensions
        (l_k, d_k, h_k, w_k) = self.kernel_size

        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.pool3d_layers = torch.nn.ModuleList()

        for i in range(l_k):


            # Initialize a Conv3D layer
            pool3d_layer = torch.nn.AvgPool3d(kernel_size=(d_k, h_k, w_k),
                                           stride=(self.stride[1], self.stride[2], self.stride[3]),
                                           padding=(self.padding))

            # Store the layer
            self.pool3d_layers.append(pool3d_layer)

    # -------------------------------------------------------------------------

    def forward(self, input):

        # Define shortcut names for dimensions of input and kernel
        (b, c_i, l_i, d_i, h_i, w_i) = tuple(input.shape)
        (l_k, d_k, h_k, w_k) = self.kernel_size

        # Compute the size of the output tensor based on the zero padding
        (l_o, d_o, h_o, w_o) = (l_i + 2 * self.padding - l_k + 1,
                                d_i + 2 * self.padding - d_k + 1,
                                h_i + 2 * self.padding - h_k + 1,
                                w_i + 2 * self.padding - w_k + 1)

        # Output tensors for each 3D frame
        frame_results = l_o * [None]
        Stride_Temporär = self.stride[0]
        # Convolve each kernel frame i with each input frame j
        for i in range(l_k):
            Count = 0 # counter for temporal Stride  
            for j in range(l_i):
                # Add results to this output frame
                out_frame = j - (i - l_k // 2) - (l_i - l_o) // 2
                if (out_frame < 0 or out_frame >= l_o):
                    continue
                if (Count % Stride_Temporär == 0):   #Apply Temporal Stride here
                    # convolve input frame j with kernel frame i
                    frame_pool3d = \
                        self.pool3d_layers[i](input[:, :, j, :]
                                            .view(b, c_i, d_i, h_i, w_i))
                    Count = Count +  1
                    # Add Results of Convolutions with the same output frame
                    if frame_results[out_frame] is None:
                        frame_results[out_frame] = frame_pool3d
                    else:
                        frame_results[out_frame] += frame_pool3d
                else:
                    Count = Count +  1

        output = torch.stack(frame_results[0::self.stride[0]], dim=2) #Stack all Resuls into the temporal Dimension (axis=2)
        output = output / l_k #Scale for Average for temporal Window 
        return output

class InstanceNorm4d(torch.nn.modules.instancenorm._InstanceNorm):
    r"""Applies Instance Normalization over a 6D input (a mini-batch of 3D inputs
    with additional channel dimension) as described in the paper
    `Instance Normalization: The Missing Ingredient for Fast Stylization`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension separately
    for each object in a mini-batch. :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size C (where C is the input size) if :attr:`affine` is ``True``.

    By default, this layer uses instance statistics computed from input data in
    both training and evaluation modes.

    If :attr:`track_running_stats` is set to ``True``, during training this
    layer keeps running estimates of its computed mean and variance, which are
    then used for normalization during evaluation. The running estimates are
    kept with a default :attr:`momentum` of 0.1.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momemtum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    .. note::
        :class:`InstanceNorm3d` and :class:`LayerNorm` are very similar, but
        have some subtle differences. :class:`InstanceNorm3d` is applied
        on each channel of channeled data like 3D models with RGB color, but
        :class:`LayerNorm` is usually applied on entire sample and often in NLP
        tasks. Additionaly, :class:`LayerNorm` applies elementwise affine
        transform, while :class:`InstanceNorm3d` usually don't apply affine
        transform.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for batch normalization.
            Default: ``False``.
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples::

        >>> # Without Learnable Parameters
        >>> m = nn.InstanceNorm3d(100)
        >>> # With Learnable Parameters
        >>> m = nn.InstanceNorm3d(100, affine=True)
        >>> input = torch.randn(20, 100, 35, 45, 10)
        >>> output = m(input)

    .. _`Instance Normalization: The Missing Ingredient for Fast Stylization`:
        https://arxiv.org/abs/1607.08022
    """

    def _check_input_dim(self, input):
        if input.dim() != 6:
            raise ValueError('expected 6D input (got {}D input)'
                             .format(input.dim()))

class _NormBase(torch.nn.modules.Module):
    """Common base of _InstanceNorm and _BatchNorm"""
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked',
                     'num_features', 'affine']

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_NormBase, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

class _BatchNorm(_NormBase):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        # Save input dimension
        input_shape = input.shape
        # Reshape to 2D 
        input = input.view(input.shape[0],input.shape[1],-1)
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        batch_out = F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        
        return batch_out.view(input_shape)

class BatchNorm4d(_BatchNorm):
    r"""Applies Batch Normalization over a 6D input (a mini-batch of 1D
    inputs with optional additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm1d(100, affine=False)
        >>> input = torch.randn(20, 100)
        >>> output = m(input)

    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167
    """

    def _check_input_dim(self, input):
        if input.dim() != 6:
            raise ValueError('expected 6D input (got {}D input)'
                             .format(input.dim()))



# -----------------------------------------------------------------------------
# MAIN CODE (TO TEST CONV4D)
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    print()
    print('TEST PYTORCH CONV4D LAYER IMPLEMENTATION')
    print('\n' + 80 * '-' + '\n')

    # -------------------------------------------------------------------------
    # Generate random input 4D tensor (+ batch dimension, + channel dimension)
    # -------------------------------------------------------------------------

    np.random.seed(42)

    input_numpy = np.round(np.random.random((1, 1, 10, 11, 12, 13)) * 100)
    input_torch = torch.from_numpy(input_numpy).float()

    # -------------------------------------------------------------------------
    # Convolve with a randomly initialized kernel
    # -------------------------------------------------------------------------

    print('Randomly Initialized Kernels:\n')

    # Initialize the 4D convolutional layer with random kernels
    conv4d_layer = \
        Conv4d(in_channels=1,
               out_channels=1,
               kernel_size=(3, 3, 3, 3),
               bias_initializer=lambda x: torch.nn.init.constant_(x, 0))

    # Pass the input tensor through that layer
    output = conv4d_layer.forward(input_torch).data.numpy()

    # Select the 3D kernels for the manual computation and comparison
    kernels = [conv4d_layer.conv3d_layers[i].weight.data.numpy().flatten()
               for i in range(3)]

    # Compare the conv4d_layer result and the manual convolution computation
    # at 3 randomly chosen locations
    for i in range(3):

        # Randomly choose a location and select the conv4d_layer output
        loc = [np.random.randint(0, output.shape[2] - 2),
               np.random.randint(0, output.shape[3] - 2),
               np.random.randint(0, output.shape[4] - 2),
               np.random.randint(0, output.shape[5] - 2)]
        conv4d = output[0, 0, loc[0], loc[1], loc[2], loc[3]]

        # Select slices from the input tensor and compute manual convolution
        slices = [input_numpy[0, 0, loc[0] + j, loc[1]:loc[1] + 3,
                              loc[2]:loc[2] + 3, loc[3]:loc[3] + 3].flatten()
                  for j in range(3)]
        manual = np.sum([slices[j] * kernels[j] for j in range(3)])

        # Print comparison
        print(f'At {tuple(loc)}:')
        print(f'\tconv4d:\t{conv4d}')
        print(f'\tmanual:\t{manual}')

    print('\n' + 80 * '-' + '\n')

    # -------------------------------------------------------------------------
    # Convolve with a kernel initialized to be all ones
    # -------------------------------------------------------------------------

    print('Constant Kernels (all 1):\n')

    conv4d_layer = \
        Conv4d(in_channels=1,
               out_channels=1,
               kernel_size=(3, 3, 3, 3),
               padding=1,
               kernel_initializer=lambda x: torch.nn.init.constant_(x, 1),
               bias_initializer=lambda x: torch.nn.init.constant_(x, 0))
    output = conv4d_layer.forward(input_torch)

    # Define relu(x) = max(x, 0) for simplified indexing below
    def relu(x: float) -> float:
        return x * (x > 0)

    # Compare the conv4d_layer result and the manual convolution computation
    # at 3 randomly chosen locations
    for i in range(3):

        # Randomly choose a location and select the conv4d_layer output
        loc = [np.random.randint(0, output.shape[2] - 2),
               np.random.randint(0, output.shape[3] - 2),
               np.random.randint(0, output.shape[4] - 2),
               np.random.randint(0, output.shape[5] - 2)]
        conv4d = output[0, 0, loc[0], loc[1], loc[2], loc[3]]

        # For a kernel that is all 1s, we only need to sum up the elements of
        # the input (the ReLU takes care of the padding!)
        manual = input_numpy[0, 0,
                             relu(loc[0] - 1):loc[0] + 2,
                             relu(loc[1] - 1):loc[1] + 2,
                             relu(loc[2] - 1):loc[2] + 2,
                             relu(loc[3] - 1):loc[3] + 2].sum()

        # Print comparison
        print(f'At {tuple(loc)}:')
        print(f'\tconv4d:\t{conv4d}')
        print(f'\tmanual:\t{manual}')

    print()