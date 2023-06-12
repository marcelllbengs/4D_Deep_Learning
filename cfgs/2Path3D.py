import os
import sys
import h5py
import re
import csv
import numpy as np



def init(mdlParams):
    ### Model Selection ###
    mdlParams['model_type'] = 'nPath_NetnD' 
    mdlParams['normalization'] = 'BatchNorm'
    mdlParams['numOut'] = 3
    mdlParams['spatial_dims'] = '3D'  # use 3D network
    mdlParams['start_to_end_estimation'] = True # 2-Path-Network (NI)
    
    ### Network Parameters ###
    mdlParams['num_init_features'] = 14 # number of init. features for the first conv. layers
    mdlParams['Dense_Growth'] = 10  # growth rate 
    mdlParams['block_config'] = (3,3,3,3) # number of layers in each densenetblock
    mdlParams['conv_last'] = False # replace last densenet block with a simple conv. with s=2.

    mdlParams['Dense_Compress'] = 1.25 # divide by this number in transition layer
    mdlParams['init_kernel'] = (3,3,3) # kernel size of the init. conv layer
    mdlParams['init_stride'] = (1,1,1) # stride of the init. conv layer
    mdlParams['spatial_padding'] = int((mdlParams['init_kernel'][1] - 1) / 2) # same padding
    mdlParams['factor_features'] = 2 # feature scaling at the fusion point of the multi-path network
    mdlParams['conv_pooling'] = False # use a conv. layer instead of an avg. pooling layer for the transition blocks
    
    mdlParams['t_downsample'] = 1 # in transition layer of densenet (stride)
    mdlParams['t_downsample_kernel'] = 1 # kernel size for transition layer
    
    ########################### Select Data  ###########################
    mdlParams['Data_Type'] = '4D'
    mdlParams['input_size'] = [32,32,32] # input size along the spatial dimension
    mdlParams['Temporal_Window'] = 2 # temporal dimension, i.e., sequence length
  
    return mdlParams



