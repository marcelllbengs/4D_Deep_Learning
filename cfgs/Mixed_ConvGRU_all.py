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

    ### Model Selection ###
    mdlParams['model_type'] = 'MixedCNN_convGRU_CNN' #Dense121
    mdlParams['convRNN_position'] = 'all'

    mdlParams['normalization'] = 'BatchNorm'
    mdlParams['numOut'] = 3
    mdlParams['many_to_many'] = True
    mdlParams['multi_output_m2many'] = True
    mdlParams['cg_type'] = 'GRU'

    mdlParams['Temporal_Window'] = 50   
    mdlParams['Temporal_Window_eval'] = 280 
    mdlParams['conv_transition'] = False

    ### Network Parameters ###
    mdlParams['cg_layers'] = [14]
    mdlParams['cg_batchnorm'] = False 
    # Grwothrate per layer
    mdlParams['num_init_features'] = 10
    mdlParams['Dense_Growth'] = 10 
    # Number of dense layers per block
    mdlParams['block_config'] = 2 
    # Compression rate at transition layers
    mdlParams['Dense_Compress'] = 1.25 # divide by this number in transition layer
    # Dimension of Networks Input (also changes the dimension of the network)
    mdlParams['spatial_dims'] = '3D' 

    mdlParams['init_kernel'] = (3,3,3) # was 7,7 before
    mdlParams['init_stride'] = (2,2,2)    
    mdlParams['spatial_padding'] = int((mdlParams['init_kernel'][1] - 1) / 2) # same padding
    
    ########################### Select Data  ###########################
    # 3D_MS = Mean + Std in channel
    mdlParams['Data_Type'] = '4D'
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
    mdlParams['batchSize'] =  5
    mdlParams['batchSize_eval'] = 1
    mdlParams['training_steps'] = 70 # 70 number of epochs for training

    # Initial learning rate
    mdlParams['learning_rate'] = 0.001/2
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
    mdlParams['mean_std_normalize_individual']  = True 
    mdlParams['label_standardization'] = False
    mdlParams['label_normalization'] = False
    
    mdlParams['MSE_Loss'] = False
    mdlParams['MAE_Loss'] = True

    # CL adpation
    mdlParams['curriculum_learning'] = True
    mdlParams['n_epoch_list'] = [10,20,30,40,50,60,70,80]
    mdlParams['n_t_list'] =     [50,50,150,150,280,280,280]

    mdlParams['adapt_batch_size_during_training'] = False
    mdlParams['16_bit_train'] = True

    mdlParams['TBTT'] = True # use TBTCC Training
    mdlParams['TBTT_t_window_steps'] = [10,20,30,40,50,60,70,80]
    mdlParams['TBTT_t_window'] =       [50,50,50,50,50,50,50,50]
    mdlParams['TBTT_starting_epochs'] = 20  
    print('Use 16-bit Training', mdlParams['16_bit_train'])
    
    return mdlParams


