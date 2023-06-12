import os
import sys
import h5py
import re
import csv
import numpy as np



def init(mdlParams):
    ### Model Selection ###
    mdlParams['model_type'] = 'MixedCNN_convGRU_CNN' 
    mdlParams['convRNN_position'] = 'all' # type of MixedConvGRU (front,middle,end,all)
    mdlParams['normalization'] = 'BatchNorm'
    mdlParams['numOut'] = 3
    mdlParams['spatial_dims'] = '3D'  # use 3D network, i.e., 3DCNN 

    mdlParams['many_to_many'] = True # many-to- many RNN
    mdlParams['multi_output_m2many'] = True # multi-output regression
    mdlParams['cg_type'] = 'GRU' # type of ConvRNN GRU or LSTm
    mdlParams['conv_transition'] = False # use a conv. layer instead of an avg. pooling layer for the transition blocks

    ### Network Parameters ###
    mdlParams['cg_layers'] = [14] # number of feature maps for the first ConvRNN
    mdlParams['cg_batchnorm'] = False # use batch norm for ConvRNN
    mdlParams['num_init_features'] = 10 # number of init. features for the first conv. layers
    mdlParams['Dense_Growth'] = 10 # growth rate 
    mdlParams['block_config'] = 2 # number of layers in each densenetblock
    mdlParams['Dense_Compress'] = 1.25 # divide by this number in transition layer

    mdlParams['init_kernel'] = (3,3,3) # kernel size of the init. conv layer
    mdlParams['init_stride'] = (2,2,2) # stride of the init. conv layer
    mdlParams['spatial_padding'] = int((mdlParams['init_kernel'][1] - 1) / 2) # same padding
    
    ########################### Select Data  ###########################
    mdlParams['Data_Type'] = '4D'       
    mdlParams['input_size'] = [32,32,32] # input size along the spatial dimension
    mdlParams['Temporal_Window'] = 50 # sequence length used during training
    mdlParams['Temporal_Window_eval'] = 280 # sequence length during evaluation

    return mdlParams


