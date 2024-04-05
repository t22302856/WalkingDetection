#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 4 2024

@author: kai-chunliu
"""
import numpy as np
import pandas as pd
import CrossValidation_SpecifyActivityTrain_ADLTest_simpleversion as CrossValidation_SpecifyActivityTrain_ADLTest
import os
from utilities import PerformanceSave
import time
import hashlib



#/home/kaichunliu_umass_edu/AHHA/SHAlab/Preprocessing/agOnly/
#/home/kaichunliu_umass_edu/AHHA/SHAlab/ADL/
#/home/kaichunliu_umass_edu/ssl-wearables-main/model_check_point/mtl_best.mdl
config = {
    'LabelTarget': 2, #1: 14 classes, 2: Binary-Classes, 3: 4 classses, 4: 5 classes, 5: 4classes
    'FS':30, # sampling rate
    'window_size': 300,
    'overlap': 30,
    'fold': 3,
    'cur_fold':1,
    'RunTimes': 5,
    'cur_RunTimes':-1,    
    'nb_layers': 2,
    'nb_channels': 3,
    'nb_classes': 2,
    'conv_filters': 8,
    'fc_filters': 8,
    'filter_width': 7,
    'max1d_width': 2,
    'drop_prob': 0.2,
    'resnet_version': 1,
    'load_weights': False,
    'freeze_weight': False,
    'seed': 1,
    'epochs': 2, #50
    'batch_size': 64,
    'learning_rate': 1e-4,
    'weight_decay': 1e-6,
    'print_counts': False,
    'early_stopping': True,
    'patience': 5,
    'SpecifiedActivity_train':False,
    'ADL_train':True,
    'LoosFunction': 'normal',
    'ModelName': 'ResNet', #CNNc3f1 ResNet
    'Plot_Flag': True,
    'load_path':'./Lab/', #lab-based data
    'load_path_ADL': './ADL/', # ADL data
    'flip_net_path': './model_check_point/mtl_best.mdl', # pre-trained model
    'output_path': 'outputs/',
}

def main(config):
    
    # load label data & ADL data
    df=pd.read_csv(config['load_path']+'Data_table.csv')
    df_r = df.loc[df['hasAG']==1].iloc[:20,:].reset_index(drop=True)
    df_ADL = pd.read_csv(config['load_path_ADL']+'/Data_table.csv')
    
    # create output folder
    t = time.localtime()
    current_time = time.strftime("%m%d%y_%H%M%S_", t)
    config['output_path'] = current_time + config['output_path']
    try:
        os.mkdir(config['output_path'])
    except:
        print(config['output_path']+'is exst')
    
    # training & testing
    Performance, Performance_subject_df, confu_AllFold = CrossValidation_SpecifyActivityTrain_ADLTest.LSOCV(df_r, df_ADL, config)
    PerformanceSave(config, Performance, Performance_subject_df, confu_AllFold)
    
if __name__ == '__main__':

    config['output_path'] = f"{config['ModelName']}_outputs/"
    main(config)

                
            

