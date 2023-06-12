import os
import torch
from torch import nn
from torch.autograd import Variable

# code adapted from / credits to https://github.com/happyjin/ConvGRU-pytorch

# Copyright (c) 2019 Jin Li

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, padding, bias, dtype, dimensionality=2, batch_norm=False, layer_norm=False):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.height, self.width, self.depths = input_size
        self.dimensionality = dimensionality
        self.kernel_size = kernel_size
        self.padding = padding
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        #print("input dim",input_dim,"hidden dim",self.hidden_dim)

        if self.layer_norm:
            self.bias = False

            # output (z)
            self.conv_z_i = self.convxd(in_channels=input_dim, 
                                        out_channels=self.hidden_dim,  # for update_gate,reset_gate respectively
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        bias=self.bias)

            self.conv_z_h = self.convxd(in_channels=self.hidden_dim, 
                                        out_channels=self.hidden_dim,  # for update_gate,reset_gate respectively
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        bias=self.bias)

            # reset (r)
            self.conv_r_i = self.convxd(in_channels=input_dim, 
                                        out_channels=self.hidden_dim,  # for update_gate,reset_gate respectively
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        bias=self.bias)

            self.conv_r_h = self.convxd(in_channels=self.hidden_dim, 
                                        out_channels=self.hidden_dim,  # for update_gate,reset_gate respectively
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        bias=self.bias)

            # output/ hidden
            self.conv_hn_x = self.convxd(in_channels=input_dim,
                                out_channels=self.hidden_dim, # for candidate neural memory
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)

            self.conv_hn_h = self.convxd(in_channels=self.hidden_dim,
                                out_channels=self.hidden_dim, # for candidate neural memory
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)


            self.ln_z_i = torch.nn.LayerNorm([self.hidden_dim, self.height, self.width, self.depths], elementwise_affine=False)
            self.ln_z_h = torch.nn.LayerNorm([self.hidden_dim, self.height, self.width, self.depths], elementwise_affine=False)
            self.ln_r_i = torch.nn.LayerNorm([self.hidden_dim, self.height, self.width, self.depths], elementwise_affine=False)
            self.ln_r_h = torch.nn.LayerNorm([self.hidden_dim, self.height, self.width, self.depths], elementwise_affine=False)
            self.ln_hn_x = torch.nn.LayerNorm([self.hidden_dim, self.height, self.width, self.depths], elementwise_affine=False)
            self.ln_hn_h = torch.nn.LayerNorm([self.hidden_dim, self.height, self.width, self.depths], elementwise_affine=False)

        else:
            self.conv_gates = self.convxd(in_channels=input_dim + hidden_dim,
                                        out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        bias=self.bias)

            self.conv_can = self.convxd(in_channels=input_dim+hidden_dim,
                                out_channels=self.hidden_dim, # for candidate neural memory
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)

            if self.batch_norm:   
                if self.dimensionality == 2:
                    self.batch_norm_layer_in = torch.nn.BatchNorm2d(input_dim)
                elif self.dimensionality == 3:                         
                    self.batch_norm_layer_in = torch.nn.BatchNorm3d(input_dim)

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width, self.depths)).type(self.dtype))

    def forward(self, input_tensor, h_cur):
        """

        :param self:
        :param input_tensor: (b, c, h, w, d)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w, d)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        #print("input",input_tensor.shape,"h_cur",h_cur.shape)
        if self.layer_norm:
            out_z_i = self.conv_z_i(input_tensor)
            out_r_i = self.conv_r_i(input_tensor)
            out_z_h = self.conv_z_h(h_cur)
            out_r_h = self.conv_r_h(h_cur)
            # apply layer norm
            out_z_i = self.ln_z_i(out_z_i) 
            out_r_i = self.ln_r_i(out_r_i)
            out_z_h = self.ln_z_h(out_z_h)
            out_r_h = self.ln_r_h(out_r_h)
            # sigmoid and sum
            z = torch.sigmoid(out_z_h) +  torch.sigmoid(out_z_i)
            r = torch.sigmoid(out_r_h) +  torch.sigmoid(out_r_i)
            # get hnew
            # input part
            out_hn_i = self.conv_hn_x(input_tensor)
            out_hn_i = self.ln_hn_x(out_hn_i) 
            # previous state part
            out_hn_h = self.conv_hn_h(h_cur)
            out_hn_h = self.ln_hn_h(out_hn_h)
            hn = out_hn_i + r*out_hn_h
            hn = torch.tanh(hn)
            h_next = (1-z)*h_cur + z*hn

        else:
            if self.batch_norm: 
                input_tensor = self.batch_norm_layer_in(input_tensor)
            combined = torch.cat([input_tensor, h_cur], dim=1)
            combined_conv = self.conv_gates(combined)
            #print("combined_conv",combined_conv.shape)

            gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)

            if self.layer_norm: # apply layernorm on gamma and beta
                gamma = self.ln_cell_gamma(gamma)
                beta = self.ln_cell_beta(gamma)

            reset_gate = torch.sigmoid(gamma)
            update_gate = torch.sigmoid(beta)

            combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
            #print("combined after",combined.shape)
            cc_cnm = self.conv_can(combined)
            #print("combined after conv",cc_cnm.shape)

            if self.layer_norm: # apply layernorm
                cc_cnm = self.ln_cell_2(cc_cnm)

            cnm = torch.tanh(cc_cnm)

            h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next

    def convxd(self,in_channels,out_channels,kernel_size,stride=1,padding=0,bias=False):
        return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)  

class ConvGRU(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, padding, num_layers,
                 dtype, batch_first=True, bias=True, return_all_layers=False, dimensionality=3, batch_norm=False, layer_norm = False, t_skip_first = False, dialation = 0):
        """

        :param input_size: (int, int, int)
            Height and width and depth of input tensor as (height, width, depth).
        :param input_dim: int e.g. 256
            Number of channels of input tensor.
        :param hidden_dim: int e.g. 1024
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param num_layers: int
            Number of ConvLSTM layers
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        :param batch_first: bool
            if the first position of array is batch or not
        :param bias: bool
            Whether or not to add the bias.
        :param return_all_layers: bool
            if return hidden and cell states for all layers
        """
        super(ConvGRU, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width, self.depths = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.dtype = dtype
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.dimensionality = dimensionality
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.t_skip_first = t_skip_first
        self.dialation = dialation

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            if self.t_skip_first == True:
                cur_input_dim = cur_input_dim + self.hidden_dim[i]

            cell_list.append(ConvGRUCell(input_size=(self.height, self.width, self.depths),
                                         input_dim = cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         padding=self.padding[i],
                                         bias=self.bias,
                                         dtype=self.dtype,
                                         dimensionality=self.dimensionality,
                                         batch_norm=self.batch_norm,
                                         layer_norm = self.layer_norm))

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        :param input_tensor: (b, t, c, h, w, d) or (t,b,c,h,w, d) depends on if batch first or not
            extracted features from alexnet
        :param hidden_state:
        :return: layer_output_list, last_state_list
        """
        if not self.batch_first:
            if self.dimensionality == 3:
                # (t, b, c, h, w, d) -> (b, t, c, h, w, d)
                input_tensor = input_tensor.permute(1, 0, 2, 3, 4, 5)
            elif self.dimensionality == 2:
                # (t, b, c, h, w) -> (b, t, c, h, w)
                input_tensor = input_tensor.permute(1, 0, 2, 3, 4)                

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            test = True
            hidden_state = [hidden_state]  
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            h_init = h

            output_inner = []
            for t in range(seq_len):
                # input current hidden and cell state then compute the next hidden and cell state through ConvLSTMCell forward function
                if self.dimensionality == 3:
                    if self.t_skip_first == True: # skip connection of the first state to all other states
                        if t == 0:
                            input_skip = cur_layer_input[:, t, :, :, :, :]
                            #print(input_skip.size())
                            #print(h.size())
                            i_z = h # take the inital hidden state as input - should be zero?
                            #print(i_z) 
                            input_skip = torch.cat([input_skip, i_z], 1)
                        else: 
                            input_skip = cur_layer_input[:, t, :, :, :, :]
                            input_skip = torch.cat([input_skip, h_0], 1)
                    else:
                        input_skip = cur_layer_input[:, t, :, :, :, :]

                    h = self.cell_list[layer_idx](input_tensor= input_skip, # (b,t,c,h,w,d)
                                              h_cur=h)

                elif self.dimensionality == 2:
                    h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], # (b,t,c,h,w)
                                              h_cur=h)

                if t == 0 and self.t_skip_first == True: 
                    h_0 = h

                output_inner.append(h)
                if self.dialation !=0:
                    if t + 1 - self.dialation > 0:
                        h = output_inner[-self.dialation]
                    else:
                        h = h_init

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            # Avg pooling layer here? 

            layer_output_list.append(layer_output)
            last_state_list.append([h])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == '__main__':
    # set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # detect if CUDA is available or not
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        dtype = torch.cuda.FloatTensor # computation in GPU
    else:
        dtype = torch.FloatTensor

    height = width = 6
    channels = 256
    hidden_dim = [32, 64]
    kernel_size = [3] # kernel size for two stacked hidden layer
    padding = [1]
    num_layers = 2 # number of stacked hidden layer
    model = ConvGRU(input_size=(height, width),
                    input_dim=channels,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    num_layers=num_layers,
                    dtype=dtype,
                    batch_first=True,
                    bias = True,
                    return_all_layers = False)

    batch_size = 1
    time_steps = 1
    input_tensor = torch.rand(batch_size, time_steps, channels, height, width)  # (b,t,c,h,w)
    layer_output_list, last_state_list = model(input_tensor)