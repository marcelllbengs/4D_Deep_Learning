import os
import sys
import h5py
import re
import csv
import numpy as np

from glob import glob
import scipy
import pickle

from natsort import natsorted
from pathlib import Path



def init(mdlParams):
    #mdlParams = {}
    # Number of GPUs on device
    mdlParams['numGPUs'] = [0]

    ### Model Selection ###
    mdlParams['model_type'] = 'nPath_NetnD' #Dense121
    mdlParams['normalization'] = 'BatchNorm'
    mdlParams['numOut'] = 3
    mdlParams['spatial_dims'] = '3D'

    mdlParams['start_to_end_estimation'] = True
    mdlParams['start_to_end_estimation_variable_start'] = True
    
    ### Network Parameters ###
    # Grwothrate per layer
    mdlParams['num_init_features'] = 14
    mdlParams['Dense_Growth'] = 10 
    # Number of dense layers per block
    mdlParams['block_config'] = (3,3,3,3) # 3,5,5 before
    mdlParams['conv_last'] = False # replace last densenet block with a simple conv. with s=2.

    # Compression rate at transition layers
    mdlParams['Dense_Compress'] = 1.25 # divide by this number in transition layer

    mdlParams['init_kernel'] = (3,3,3) # was 7,7 before
    mdlParams['init_stride'] = (1,1,1)    
    mdlParams['spatial_padding'] = int((mdlParams['init_kernel'][1] - 1) / 2) # same padding
    mdlParams['factor_features'] = 2
    mdlParams['conv_pooling'] = False
    
    mdlParams['t_downsample'] = 1
    mdlParams['t_downsample_kernel'] = 1
    
    ########################### Select Data  ###########################
    mdlParams['Data_Type'] = '4D'
    mdlParams['Temporal_Window'] = 2
    # directory to the data
    mdlParams['dataDir'] = mdlParams['pathBase'] + '/ultrasoundh5/' 
    mdlParams['all_files_path'] = mdlParams['dataDir']  

    ########################## Set Index for Train, Test, Val ##########################

    # Set up Train and Validation Index
    mdlParams['trainInd'] = [] 
    mdlParams['valInd'] = [] 
    mdlParams['testInd'] = [] 

    # Set Index for Folds
    mdlParams['numCV'] = 1
    mdlParams['trainIndCV'] = []
    mdlParams['valIndCV'] = []
    mdlParams['testIndCV'] = []
        
    # set up the different indices (here mock indices)
    mdlParams['valInd'] = range(1, 10)
    mdlParams['testInd'] = range(10, 20)
    mdlParams['trainInd'] = range(20, 30)
    mdlParams['trainInd_eval'] = range(20, 30)
    mdlParams['trainInd_evalCV'] = []

    # set up data split
    print('Validation (number of different trajectories)',  len(mdlParams['valInd'])) 
    print('Test (number of different trajectories)',  len(mdlParams['testInd']))
    print('Train (number of different trajectories)',  len(mdlParams['trainInd']))

    mdlParams['valIndCV'].append(np.array(mdlParams['valInd'])) # save the current Validation indices
    mdlParams['testIndCV'].append(np.array(mdlParams['testInd'])) # save the current Validation indices
    mdlParams['trainIndCV'].append(np.array(mdlParams['trainInd'] ))  # set up the corresponding train set
    mdlParams['trainInd_evalCV'].append(np.array(mdlParams['trainInd'][::50]))

    # Ind properties
    print("Train")
    for i in range(len(mdlParams['trainIndCV'])):
        print(mdlParams['trainIndCV'][i].shape)
    print("Val")
    for i in range(len(mdlParams['valIndCV'])):
        print(mdlParams['valIndCV'][i].shape)      
        print("Intersect",np.intersect1d(mdlParams['trainIndCV'][i],mdlParams['valIndCV'][i]))   
    print("test")
    for i in range(len(mdlParams['testIndCV'])):
        print(mdlParams['testIndCV'][i].shape)      
    print("Intersect",np.intersect1d(mdlParams['trainIndCV'][i],mdlParams['testIndCV'][i]))        

    ######################### Set Up Labels for Data Set ##########################
    # get max and min for data scaling of training data labels 
    mdlParams['Max_Data_Scale'] = [1, 1, 1]
    mdlParams['Min_Data_Scale'] = [0, 0, 0]
    print(mdlParams['Max_Data_Scale'])
    print(mdlParams['Min_Data_Scale'])

    # # Input size          
    mdlParams['input_size'] = [32,32,32]
    mdlParams['print_trainerr'] = True
    ### Training Parameters ###
    # Batch size
    mdlParams['batchSize'] = 10
    mdlParams['batchSize_eval'] = mdlParams['batchSize'] 
    mdlParams['training_steps']  = 70

    # Initial learning rate
    mdlParams['learning_rate'] = 0.001*len(mdlParams['numGPUs'])
    # Lower learning rate after no improvement over 100 epochs
    mdlParams['lowerLRAfter'] = 10000 # was 25 before
    # If there is no validation set, start lowering the LR after X steps
    mdlParams['lowerLRat'] = 10000
    # Divide learning rate by this value
    mdlParams['LRstep'] = 1 #changed from 2 to 10 because of poor convergence during traning
    # Display error every X steps
    mdlParams['display_step'] = 5
    # Print trainerr
    mdlParams['print_trainerr'] = True
    mdlParams['use_test_set'] = True
    # Subtract trainset mean?
    mdlParams['subtract_set_mean'] = False
    mdlParams['mean_std_normalize']  = True 
    mdlParams['label_standardization'] = False
    mdlParams['label_normalization'] = False
    mdlParams['MSE_Loss'] = False
    mdlParams['MAE_Loss'] = True
    mdlParams['16_bit_train'] = True

    return mdlParams



