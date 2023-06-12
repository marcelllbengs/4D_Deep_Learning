import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import numpy as np
import functools
import h5py
import math
import os
import convGRU
import convLSTM
from collections import OrderedDict

from conv4d import Conv4d
from conv4d import Pool4d
from conv4d import BatchNorm4d
from conv4d import InstanceNorm4d
from einops import rearrange, repeat
from einops import rearrange, repeat


# DenseNet is implemented based on https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016, 
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

def Normalization(n_feat,mdlParams):

    if mdlParams['normalization'] == 'BatchNorm':
        out = BatchNorm4d(n_feat)
    elif mdlParams['normalization'] == 'InstanceNorm':
        out = InstanceNorm4d(n_feat)
    else:
        out = n_feat

    return out

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate,dimension,mdlParams):
        super(_DenseLayer, self).__init__()
        
        if mdlParams.get('CNN_batchnorm','BatchNorm') ==  'BatchNorm':
            use_bias = False
        else: 
            use_bias = True

        if dimension == '3D':
            if mdlParams.get('CNN_batchnorm','BatchNorm') ==  'BatchNorm':
                self.add_module('norm1', nn.BatchNorm3d(num_input_features)),

            self.add_module('relu1', nn.ReLU(inplace=True)),
            self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                            growth_rate, kernel_size=1, stride=1, bias=use_bias)),

            if mdlParams.get('CNN_batchnorm','BatchNorm') ==  'BatchNorm':
                self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),

            self.add_module('relu2', nn.ReLU(inplace=True)),
            self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                            kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=use_bias)),
            self.drop_rate = drop_rate

        elif dimension == '4D':
            if mdlParams.get('CNN_batchnorm','BatchNorm') ==  'BatchNorm':
                self.add_module('norm1', Normalization(num_input_features,mdlParams)),
            self.add_module('relu1', nn.ReLU(inplace=True)),
            self.add_module('conv1', Conv4d(num_input_features, bn_size *
                            growth_rate, kernel_size=(1,1,1,1), stride=(1,1,1,1), bias=use_bias)),
            if mdlParams.get('CNN_batchnorm','BatchNorm') ==  'BatchNorm':
                self.add_module('norm2', Normalization(bn_size * growth_rate,mdlParams)),
            self.add_module('relu2', nn.ReLU(inplace=True)),
            self.add_module('conv2', Conv4d(bn_size * growth_rate, growth_rate,
                                kernel_size=(3,3,3,3), stride=(1,1,1,1), padding=(1,1,1,1), bias=use_bias)),
            self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseLayer_v2(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate,dimension,mdlParams):
        super(_DenseLayer_v2, self).__init__()
        
        if mdlParams.get('CNN_batchnorm','BatchNorm') ==  'BatchNorm':
            use_bias = False
        else: 
            use_bias = True

        if dimension == '3D':
            
            self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                            growth_rate, kernel_size=1, stride=1, bias=use_bias)),
            if mdlParams.get('CNN_batchnorm','BatchNorm') ==  'BatchNorm':
                self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
            self.add_module('relu2', nn.ReLU(inplace=True)),
            self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                            kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=use_bias)),
            self.drop_rate = drop_rate

        elif dimension == '4D':
            self.add_module('conv1', Conv4d(num_input_features, bn_size *
                            growth_rate, kernel_size=(1,1,1,1), stride=(1,1,1,1), bias=use_bias)),
            if mdlParams.get('CNN_batchnorm','BatchNorm') ==  'BatchNorm':
                self.add_module('norm2', Normalization(bn_size * growth_rate,mdlParams)),
            self.add_module('relu2', nn.ReLU(inplace=True)),
            self.add_module('conv2', Conv4d(bn_size * growth_rate, growth_rate,
                                kernel_size=(3,3,3,3), stride=(1,1,1,1), padding=(1,1,1,1), bias=use_bias)),
            self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer_v2, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock_v2(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, dimension,mdlParams):
        super(_DenseBlock_v2, self).__init__()
        for i in range(num_layers):
            if i ==0:
                layer = _DenseLayer_v2(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, dimension, mdlParams) # do not apply batch norm and relu for the first layer.
            else:
                layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, dimension, mdlParams)
            self.add_module('denselayer%d' % (i + 1), layer)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, dimension,mdlParams):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, dimension, mdlParams)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features,dimension,mdlParams):
        super(_Transition, self).__init__()

        if dimension == '3D':
            if mdlParams.get('CNN_batchnorm','BatchNorm') ==  'BatchNorm':
                self.add_module('norm', nn.BatchNorm3d(num_input_features))
                self.add_module('relu', nn.ReLU(inplace=True))
                self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                                kernel_size=1, stride=1, bias=False))
            else:
                self.add_module('relu', nn.ReLU(inplace=True))
                self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                                kernel_size=1, stride=1, bias=True))

            if mdlParams.get('conv_pooling',False) ==  True:
                self.add_module('pool-norm', nn.BatchNorm3d(num_output_features))
                self.add_module('pool-relu', nn.ReLU(inplace=True))
                self.add_module('pool-conv', nn.Conv3d(num_output_features, num_output_features, kernel_size=3, stride=2, padding=1, bias=False)) ## replace pooling with conv here
            else:
                ## conv pooling here! 
                self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2)) ## replace pooling with conv here

        elif dimension == '4D':
            if mdlParams.get('CNN_batchnorm','BatchNorm') ==  'BatchNorm':
                self.add_module('norm', Normalization(num_input_features,mdlParams))
                self.add_module('relu', nn.ReLU(inplace=True))
                self.add_module('conv', Conv4d(num_input_features, num_output_features,
                                                kernel_size=(1,1,1,1), stride=(1,1,1,1), bias=False))
            else:
                self.add_module('relu', nn.ReLU(inplace=True))
                self.add_module('conv', Conv4d(num_input_features, num_output_features,
                                                kernel_size=(1,1,1,1), stride=(1,1,1,1), bias=True))

            if mdlParams.get('conv_pooling',False) ==  True:
                self.add_module('pool-norm', Normalization(num_output_features,mdlParams))
                self.add_module('pool-relu', nn.ReLU(inplace=True))
                t_pad =  int((mdlParams.get('t_downsample_kernel',3) - 1) / 2)  # same padding
                self.add_module('pool-conv', Conv4d(num_output_features, num_output_features, kernel_size=(mdlParams.get('t_downsample_kernel',3),3,3,3), stride=(mdlParams.get('t_downsample',2),2,2,2), padding=(t_pad,1,1,1), bias=False))
            else:
                self.add_module('pool', Pool4d(kernel_size=(mdlParams.get('t_downsample_kernel',2),2,2,2), stride=(mdlParams.get('t_downsample',2),2,2,2)))

class _Transition_conv(nn.Sequential):
    def __init__(self, num_input_features, num_output_features,dimension,mdlParams):
        super(_Transition_conv, self).__init__()

        if dimension == '3D':
            if mdlParams.get('CNN_batchnorm','BatchNorm') ==  'BatchNorm':
                self.add_module('norm', nn.BatchNorm3d(num_input_features))
                use_bias = False
            else:
                use_bias = True
            self.add_module('relu', nn.ReLU(inplace=True))

            if mdlParams.get('no_compression',False):
                num_output_features = num_input_features
            else:
                self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                                kernel_size=1, stride=1, bias=use_bias))
                
                if mdlParams.get('CNN_batchnorm','BatchNorm') ==  'BatchNorm':
                    self.add_module('norm1', nn.BatchNorm3d(num_output_features))
                self.add_module('relu1', nn.ReLU(inplace=True))

            self.add_module('pool', nn.Conv3d(num_output_features, num_output_features, kernel_size=3, stride=2, padding =1, bias=use_bias))


        elif dimension == '4D':
            if mdlParams.get('CNN_batchnorm','BatchNorm') ==  'BatchNorm':
                self.add_module('norm', Normalization(num_input_features,mdlParams))

            self.add_module('relu', nn.ReLU(inplace=True))

            if mdlParams.get('no_compression',False):
                num_output_features = num_input_features
            else:
                self.add_module('conv', Conv4d(num_input_features, num_output_features,
                                                kernel_size=(1,1,1,1), stride=(1,1,1,1), bias=False))
                if mdlParams.get('CNN_batchnorm','BatchNorm') ==  'BatchNorm':
                    self.add_module('norm1', Normalization(num_output_features,mdlParams))
                self.add_module('relu1', nn.ReLU(inplace=True))

            self.add_module('pool', Conv4d(num_output_features, num_output_features, kernel_size=(3,3,3,3), stride=(mdlParams.get('t_downsample',2),2,2,2), padding =(1,1,1,1)))


class DenseBlocknD(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, mdlParams, input_features, use_transition = True, bn_size=4, drop_rate=0, conv_transition = False):

        super(DenseBlocknD, self).__init__()

        bn_size = mdlParams.get('bn_size',4)

        self.use_transition = use_transition
        self.use_conv_transition = conv_transition
        # Each denseblock
        num_features = input_features
        num_layers = mdlParams['block_config']
        i = 0
        block = _DenseBlock_v2(num_layers=num_layers, num_input_features=num_features,
                            bn_size=bn_size, growth_rate=mdlParams['Dense_Growth'], drop_rate=mdlParams.get('dropout',0), dimension = mdlParams['spatial_dims'], mdlParams = mdlParams)
        
        #print(block)
        self.features = block
        num_features = num_features + num_layers * mdlParams['Dense_Growth']
        
        if self.use_transition == True:
            if self.use_conv_transition == False:
                trans = _Transition(num_input_features=num_features, num_output_features= int(num_features // mdlParams['Dense_Compress']),dimension = mdlParams['spatial_dims'],mdlParams = mdlParams)
            else:
                trans = _Transition_conv(num_input_features=num_features, num_output_features= int(num_features // mdlParams['Dense_Compress']),dimension = mdlParams['spatial_dims'],mdlParams = mdlParams)
            
            self.features.add_module('transition%d' % (i + 1), trans)
            num_features = int(num_features // mdlParams['Dense_Compress'])
        else:
            if mdlParams.get('CNN_batchnorm','BatchNorm') ==  'BatchNorm':
                self.features.add_module('last_batchnorm_transition', nn.BatchNorm3d(num_features)) # was missing before 23.10, mabye remove
                self.features.add_module('last_relu_transition', nn.ReLU(inplace=True)) # was missing before 23.10, maybe remove
                self.features.add_module('last_convv', nn.Conv3d(num_features, num_features, kernel_size=(3,3,3), stride= (2,2,2), padding= (1,1,1), bias=False))
            else:
                self.features.add_module('last_relu_transition', nn.ReLU(inplace=True)) # was missing before 23.10, maybe remove
                self.features.add_module('last_convv', nn.Conv3d(num_features, num_features, kernel_size=(3,3,3), stride= (2,2,2), padding= (1,1,1), bias=True))

        if mdlParams.get('CNN_batchnorm','BatchNorm') ==  'BatchNorm':
            # Final batch norm
            if mdlParams['spatial_dims'] == '2D':
                self.features.add_module('norm5', nn.BatchNorm2d(num_features))
            elif mdlParams['spatial_dims'] == '3D':
                self.features.add_module('norm5', nn.BatchNorm3d(num_features))
            elif mdlParams['spatial_dims'] == '4D':
                self.features.add_module('norm5', Normalization(num_features,mdlParams))

        # add acitvation
        self.features.add_module('relu_last', nn.ReLU(inplace=True))

        self.num_features = num_features # final number of features

        # Official init from torch repo.
        if mdlParams['spatial_dims'] == '2D':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

        elif mdlParams['spatial_dims'] == '3D':
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

        elif mdlParams['spatial_dims'] == '4D':
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #print('Input Shape: ', len(x.shape))
        features = self.features(x)
        out = features
        return out

class nPath_DenseNetnD(nn.Module):
    r"""Multi-Path DenseNetnD, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """

    def __init__(self, mdlParams, bn_size=4, drop_rate=0):

        super(nPath_DenseNetnD, self).__init__()
        
        # hyperparameters
        bn_size = mdlParams.get('bn_size',4)
        self.time_channel = mdlParams.get('time_channel',False)
        self.spatial_dims = mdlParams['spatial_dims'] 
        self.previous_vols = mdlParams.get('previous_vols_Path',2)
        self.start_to_end_estimation = mdlParams.get('start_to_end_estimation',False)
        self.drift_loss_function = mdlParams.get('drift_loss_function',False)
        self.drift_loss_n = mdlParams.get('drift_loss_n',8)
        self.conv_last = mdlParams.get('conv_last',False)

        if self.time_channel == True:
            input_features = self.previous_vols
            follow_up_factor = 1
        else:
            input_features = 1 
            follow_up_factor = self.previous_vols

        # first shared parameter path
        self.multi_path = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(input_features, mdlParams['num_init_features'], kernel_size= 3, stride =1, padding= 1, bias=False)),
            ('norm0', nn.BatchNorm3d(mdlParams['num_init_features'])),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv3d(mdlParams['num_init_features'], mdlParams['num_init_features'], kernel_size= 3, stride = 1, padding= 1, bias=False)),
            ('norm1', nn.BatchNorm3d(mdlParams['num_init_features'])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(mdlParams['num_init_features'], mdlParams['num_init_features'], kernel_size= 3, stride = 1, padding=1 , bias=False)),
            ('norm2', nn.BatchNorm3d(mdlParams['num_init_features'])),
            ('relu2', nn.ReLU(inplace=True)),
        ]))  

        ####### Follow up-network
        # First convolution
        if mdlParams['spatial_dims'] == '3D':
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv3d(mdlParams['num_init_features']*follow_up_factor,  mdlParams.get('factor_features',5)*mdlParams['num_init_features'], kernel_size=(3,3,3), stride = mdlParams.get('init_stride',1), padding= (1,1,1), bias=False)),
            ]))    
        elif mdlParams['spatial_dims'] == '4D':
            self.features = nn.Sequential(OrderedDict([
                ('conv0', Conv4d(mdlParams['num_init_features'], mdlParams.get('factor_features',5)*mdlParams['num_init_features'], kernel_size=(3,3,3,3), stride = mdlParams.get('init_stride',(1,1,1,1)), padding= (1,1,1,1), bias=False)),
            ]))  

        # Each denseblock for follow-up network
        num_features = mdlParams.get('factor_features',5)*mdlParams['num_init_features']
        for i, num_layers in enumerate(mdlParams['block_config']):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=mdlParams['Dense_Growth'], drop_rate=mdlParams.get('dropout',0), dimension = mdlParams['spatial_dims'], mdlParams = mdlParams)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * mdlParams['Dense_Growth']
            if i != len(mdlParams['block_config']) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features= int(num_features // mdlParams['Dense_Compress']),dimension = mdlParams['spatial_dims'],mdlParams = mdlParams)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features // mdlParams['Dense_Compress'])

        # Final batch norm
        if mdlParams['spatial_dims'] == '3D':
            self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        elif mdlParams['spatial_dims'] == '4D':
            self.features.add_module('norm5', Normalization(num_features,mdlParams))
        # last relu
        self.features.add_module('relu_5',  nn.ReLU(inplace=True))

        # last conv? 
        if self.conv_last ==  True:
            if mdlParams['spatial_dims'] == '3D':
                self.features.add_module('conv_last', nn.Conv3d(num_features, num_features, kernel_size=(3,3,3), stride = 2, padding= (1,1,1), bias=False))
                self.features.add_module('norm_last', nn.BatchNorm3d(num_features))
                self.features.add_module('relu_last', nn.ReLU(inplace=True))
            elif mdlParams['spatial_dims'] == '4D':
                self.features.add_module('conv_last', Conv4d(num_features, num_features, kernel_size=(3,3,3,3), stride = (1,2,2,2), padding= (1,1,1,1), bias=False))
                self.features.add_module('norm_last', Normalization(num_features,mdlParams))
                self.features.add_module('relu_last', nn.ReLU(inplace=True))

        print(self.features)

        # Linear layer
        self.classifier = nn.Linear(num_features, mdlParams['numOut'])

        if mdlParams['spatial_dims'] == '3D':
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

        elif mdlParams['spatial_dims'] == '4D':
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # stack temporal information in batch dimension
        if self.spatial_dims == '4D':
            bs = x.size(0)
            npush = x.size(2)
            for i in range(npush): 
                x_vol = x[:,:,i,:,:,:] # batch,channel,time,b,w 
                if i ==0:
                    x_new = x_vol
                else: 
                    x_new = torch.cat([x_new, x_vol], 0) # stack into batch dim t*bxcxhxw 
            # Pass through joint path
            x = x_new
            x = self.multi_path(x) # multipath processing
            x = torch.unsqueeze(x, 2) # 0(b) ,1(c) , 2(t) / add the temporal dimension

            # Restack into temp dim
            for i in range(npush):
                t_vol = x[i*bs:(i+1)*bs,:,:,:,:,:] # (batch,cxtxhxw) 
                if i ==0:
                    layer_new = t_vol
                else:
                    layer_new = torch.cat([layer_new, t_vol], 2) # temp concat/ressemble
            x = layer_new

        elif self.spatial_dims == '3D':
            if self.start_to_end_estimation == True:
                x = x[:,:,[0, -1],:,:,:] # b,c,t,h,w,d # Two-Path Network-NI
            else:
                x = x[:,:,-self.previous_vols:,:,:,:]# only keep the last n vols # Multi-Path with channel stacking approach

            if self.time_channel == True:
                x = torch.squeeze(x) # b,t,h,w,d
            else:
                bs = x.size(0)
                npush = x.size(2)
                for i in range(npush):
                    x_vol = x[:,:,i,:,:,:] # batch,channel,time,b,w 
                    if i ==0:
                        x_new = x_vol
                    else: 
                        x_new = torch.cat([x_new, x_vol], 0) # stack into batch dim t*bxcxhxw 
                # Pass through joint path
                x = x_new
            x = self.multi_path(x) # multipath processing

            if self.time_channel == False:
                # Restack into feature dimension
                for i in range(npush):
                    t_vol = x[i*bs:(i+1)*bs,:,:,:,:] # (batch,cxtxhxw) 
                    if i ==0:
                        layer_new = t_vol
                    else:
                        layer_new = torch.cat([layer_new, t_vol], 1) # temp concat/ressemble
                x = layer_new

        # now push through joint path
        out = self.features(x)
        #Take care of dimension-GAP
        if len(x.shape) == 4:
            out = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        elif len(x.shape) == 5:     
            out = F.adaptive_avg_pool3d(out, (1, 1, 1)).view(out.size(0), -1)            
        elif len(x.shape) == 6:     
            out = torch.mean(torch.mean(torch.mean(torch.mean(out,2),2),2),2) # torch.mean(torch.mean(torch.mean(torch.mean(features,2),2),2),2)

        out = self.classifier(out)

        return out

class MixedCNN_convGRU_CNN(nn.Module):

    r"""Mixed ConvGRU-architecture, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """

    
    def __init__(self, mdlParams, bn_size=4, drop_rate=0):
        super(MixedCNN_convGRU_CNN, self).__init__()  

        # hyperparameters
        self.cg_kernel_size = mdlParams.get('cg_kernel_size',[3])   
        self.cg_padding = mdlParams.get('cg_padding',[1]) 
        self.cg_layers = mdlParams.get('cg_layers',[12])  
        self.cg_batchnorm = mdlParams.get('cg_batchnorm',False)
        self.cg_layernorm = mdlParams.get('cg_layernorm',False) 
        self.cg_type = mdlParams.get('cg_type','GRU')
        self.bottleneck_factor = mdlParams.get('bottleneck_factor',4)
        self.spatial_dims = '3D'
        self.convRNN_position = mdlParams.get('convRNN_position','all')
        self.conv_transition = mdlParams.get('conv_transition',False) 
        self.dialation = mdlParams.get('dialation',[0,0,0])
        self.num_channels = 1
        self.multi_output = mdlParams.get('multi_output_m2many',False)

        # set up model
        # first conv. layers of the network
        self.features_0 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, self.cg_layers[0], kernel_size=(3,3,3), stride= (1,1,1), padding= (1,1,1), bias=False)),
            ('norm0', nn.BatchNorm3d(self.cg_layers[0])),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv3d(self.cg_layers[0], self.cg_layers[0], kernel_size=(3,3,3), stride= (1,1,1), padding= (1,1,1), bias=False)),
            ('norm1', nn.BatchNorm3d(self.cg_layers[0])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(self.cg_layers[0], self.cg_layers[0], kernel_size=(3,3,3), stride= (1,1,1), padding= (1,1,1), bias=False)),
            ('norm2', nn.BatchNorm3d(self.cg_layers[0])),
            ('relu2', nn.ReLU(inplace=True)),
        ])) 

        # Define different DenseNet blocks of the network
        self.features_1 = DenseBlocknD(mdlParams, self.cg_layers[0], use_transition = True, bn_size=self.bottleneck_factor, conv_transition = self.conv_transition) # denseblock with transition
        print(' self.features_1.num_features',self.features_1.num_features)
        print(self.features_1)
        self.features_2 = DenseBlocknD(mdlParams, self.features_1.num_features, use_transition = True, bn_size=self.bottleneck_factor, conv_transition = self.conv_transition)  # denseblock with transition
        print('self.features_2.num_features', self.features_2.num_features)
        print(self.features_2)
        self.features_3 = DenseBlocknD(mdlParams, self.features_2.num_features, use_transition = False, bn_size=self.bottleneck_factor)  # denseblock without transition
        print('self.features_3.num_features', self.features_3.num_features)
        print(self.features_3)

        # choose GRU or LSTM and define ConvRNNs modules between the blocks
        # Note the input size is hardcoded
        if self.cg_type == 'GRU':
            self.module_dict = nn.ModuleDict()   
            if self.convRNN_position =='all' or self.convRNN_position =='front':
                self.module_dict['conv_gru_1'] = convGRU.ConvGRU(input_size=(32, 32, 32),
                            input_dim = self.cg_layers[0],
                            hidden_dim = self.cg_layers,
                            kernel_size = self.cg_kernel_size,
                            padding = self.cg_padding,
                            num_layers = len(self.cg_layers),
                            dtype= torch.cuda.FloatTensor,
                            batch_first= True,
                            bias = True,
                            return_all_layers = False,
                            dimensionality = 3,
                            batch_norm= self.cg_batchnorm,
                            layer_norm= self.cg_layernorm,
                            dialation = self.dialation[0])                        
                            
            if self.convRNN_position =='all' or self.convRNN_position =='middle':
                self.module_dict['conv_gru_2'] = convGRU.ConvGRU(input_size=(16, 16, 16),
                            input_dim = self.features_1.num_features,
                            hidden_dim = [self.features_1.num_features],
                            kernel_size = self.cg_kernel_size,
                            padding = self.cg_padding,
                            num_layers = len(self.cg_layers),
                            dtype= torch.cuda.FloatTensor,
                            batch_first= True,
                            bias = True,
                            return_all_layers = False,
                            dimensionality = 3,
                            batch_norm= self.cg_batchnorm,
                            layer_norm= self.cg_layernorm,
                            dialation = self.dialation[1])

            if self.convRNN_position =='all' or self.convRNN_position =='end':
                self.module_dict['conv_gru_3'] = convGRU.ConvGRU(input_size=(8, 8, 8),
                        input_dim = self.features_2.num_features,
                        hidden_dim = [self.features_2.num_features],
                        kernel_size = self.cg_kernel_size,
                        padding = self.cg_padding,
                        num_layers = len(self.cg_layers),
                        dtype= torch.cuda.FloatTensor,
                        batch_first= True,
                        bias = True,
                        return_all_layers = False,
                        dimensionality = 3,
                        batch_norm= self.cg_batchnorm,
                        layer_norm= self.cg_layernorm,
                        dialation = self.dialation[2])
        else:
            self.module_dict = nn.ModuleDict()   
            if self.convRNN_position =='all' or self.convRNN_position =='front':
                self.module_dict['conv_gru_1'] = convLSTM.ConvLSTM(input_size=(32, 32, 32),
                            input_dim = self.cg_layers[0],
                            hidden_dim = self.cg_layers,
                            kernel_size = self.cg_kernel_size,
                            padding = self.cg_padding,
                            num_layers = len(self.cg_layers),
                            dtype= torch.cuda.FloatTensor,
                            batch_first= True,
                            bias = True,
                            return_all_layers = False,
                            dimensionality = 3,
                            batch_norm= self.cg_batchnorm)

            if self.convRNN_position =='all' or self.convRNN_position =='middle':
                self.module_dict['conv_gru_2'] = convLSTM.ConvLSTM(input_size=(16, 16, 16),
                            input_dim = self.features_1.num_features,
                            hidden_dim = [self.features_1.num_features],
                            kernel_size = self.cg_kernel_size,
                            padding = self.cg_padding,
                            num_layers = len(self.cg_layers),
                            dtype= torch.cuda.FloatTensor,
                            batch_first= True,
                            bias = True,
                            return_all_layers = False,
                            dimensionality = 3,
                            batch_norm= self.cg_batchnorm)

            if self.convRNN_position =='all' or self.convRNN_position =='end':
                self.module_dict['conv_gru_3'] = convLSTM.ConvLSTM(input_size=(8, 8, 8),
                        input_dim = self.features_2.num_features,
                        hidden_dim = [self.features_2.num_features],
                        kernel_size = self.cg_kernel_size,
                        padding = self.cg_padding,
                        num_layers = len(self.cg_layers),
                        dtype= torch.cuda.FloatTensor,
                        batch_first= True,
                        bias = True,
                        return_all_layers = False,
                        dimensionality = 3,
                        batch_norm= self.cg_batchnorm)
        
        self.classifier = nn.Linear(self.features_3.num_features, mdlParams['numOut'] )

        if mdlParams['spatial_dims'] == '3D':
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

  
    def time2batch(self, x):
        if self.spatial_dims =='3D':
            b_dim, c_dim, t_dim, h_dim, w_dim, d_dim = x.shape 
            x_new = rearrange(x, 'b_dim c_dim t_dim h_dim w_dim d_dim -> (b_dim t_dim) c_dim h_dim w_dim d_dim')
        elif self.spatial_dims =='2D':
            b_dim, c_dim, t_dim, h_dim, w_dim = x.shape 
            x_new = rearrange(x, 'b_dim c_dim t_dim h_dim w_dim -> (b_dim t_dim) c_dim h_dim w_dim')

        return x_new
    
    def batch2time(self,x,bs,seq_length):
        if self.spatial_dims =='3D':
            _, c_dim, h_dim, w_dim, d_dim = x.shape 
            layer_new = rearrange(x, '(b_dim t_dim) c_dim h_dim w_dim d_dim -> b_dim c_dim t_dim h_dim w_dim d_dim',b_dim=bs, t_dim=seq_length)
        elif self.spatial_dims =='2D':
            _, c_dim, h_dim, w_dim = x.shape
            layer_new = rearrange(x, '(b_dim t_dim) c_dim h_dim w_dim -> b_dim c_dim t_dim h_dim w_dim',b_dim=bs, t_dim=seq_length)
        return layer_new

    def forward(self, x, h_0 = None, h_1 = None, h_2 = None): 
        bs = x.size(0)
        seq_length = x.size(2)

        x = self.time2batch(x)
        x = self.features_0(x) # CNN-0

        if self.convRNN_position =='all' or self.convRNN_position =='front':
            x = self.batch2time(x,bs,seq_length)
            x =  x.permute(0,2,1,3,4,5) # batch,time,channel, h,w,d for convGRU
            # ConvGRU Network (Follow-up)
            out_list, h_0 = self.module_dict['conv_gru_1'](x,h_0)
            h_0 = h_0[-1]
            x = out_list[-1][:,:,:,:,:,:]  #  batch,time,channel, h,w,d
            x =  x.permute(0,2,1,3,4,5) # swap t and c
            x = self.time2batch(x)

        x = self.features_1(x) # CNN-1

        if self.convRNN_position =='all' or self.convRNN_position =='middle':
            x = self.batch2time(x,bs,seq_length)
            x =  x.permute(0,2,1,3,4,5) # batch,time,channel, h,w,d for convGRU
            # ConvGRU Network (Follow-up)
            out_list, h_1 = self.module_dict['conv_gru_2'](x,h_1)
            h_1 = h_1[-1]

            x = out_list[-1][:,:,:,:,:,:]  #  batch,time,channel, h,w,d
            x =  x.permute(0,2,1,3,4,5) # swap t and c
            x = self.time2batch(x)

        x = self.features_2(x) # CNN-2

        if self.convRNN_position =='all' or self.convRNN_position =='end':
            x = self.batch2time(x,bs,seq_length)
            x =  x.permute(0,2,1,3,4,5) # batch,time,channel, h,w,d for convGRU
            # ConvGRU Network (Follow-up)
            out_list, h_2 = self.module_dict['conv_gru_3'](x,h_2)
            h_2 = h_2[-1]

        # Classifier
        if self.multi_output == True:
            if self.convRNN_position =='all' or self.convRNN_position =='end':
                out = out_list[-1][:,:,:,:,:,:]  #  batch,time,channel, h,w,d
                # Stack into Batch dimension
                nout = out.size(1)
                for i in range(nout):
                    if i ==0:
                        out_new = out[:,i,:,:,:,:]
                    else: 
                        out_new = torch.cat([out_new, out[:,i,:,:,:,:]], 0) # stack into batch dimension
                out = out_new
            else:
                out = x 

            out = self.features_3(out) # CNN-Last
            out = F.adaptive_avg_pool3d(out, (1, 1, 1)).view(out.size(0), -1)        
            out = self.classifier(out)

            # Ressemble temporal dimension
            out = torch.unsqueeze(out, 1) # 0(b) ,1(c) , 2(t) / add the temporal dimension
            #print('Batch Size', bs)
            for i in range(seq_length):
                t_out = out[i*bs:(i+1)*bs,:,:] # (batch,t,numOut) 
                #print('t_out', t_out.size())
                if i ==0:
                    out_new = t_out
                else:
                    out_new = torch.cat([out_new, t_out], 1) # temp concat/ressemble
            out = out_new # (batch,t,numOut) 

        else:
            out = out_list[-1][:,-1,:,:,:,:]  # only take last temporal prediction? 
            #Take care of dimension
            if len(out.shape) == 4:
                out = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
            elif len(out.shape) == 5:     
                out = F.adaptive_avg_pool3d(out, (1, 1, 1)).view(out.size(0), -1)        
            out = self.classifier(out)
        # states = [h_0,h_1,h_2]
        if self.convRNN_position =='front':
            h_1 = [None]
            h_2 = [None]
        if self.convRNN_position =='middle':
            h_0 = [None]
            h_2 = [None]
        if self.convRNN_position =='end':
            h_0 = [None]
            h_1 = [None]

        return out, h_0[-1], h_1[-1], h_2[-1]


