#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 20:52:01 2023

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
    'window_size': 300,
    'overlap': 30,
    'post_parameter_walk': 4,
    'post_parameter_nonwalk': 4,
    'fold': 5,
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
    'ResNetBlock': 'BasicBlock',
    'ResNetLayer': [2,2,2],
    'resnet_version': 1,
    'load_weights': False,
    'freeze_weight': False,
    'seed': 1,
    'epochs': 10, #50
    'batch_size': 64,
    'learning_rate': 1e-4,
    'weight_decay': 1e-6,
    'print_counts': False,
    'early_stopping': True,
    'patience': 5,
    'SpecifiedActivity_train':False,
    'ADL_train':True,
    'LoosFunction': 'normal',
    'ModelName': 'CNNc3f1', #CNNc3f1 RseNet
    'DataAug': False,
    'load_path':'/Users/kai-chunliu/Documents/UMass/AHHA/Shirley Ryan AbilityLab/Preprocessing/agOnly/', #lab-based data
    'load_path_ADL': '/Users/kai-chunliu/Documents/UMass/AHHA/Shirley Ryan AbilityLab/Labeling/ADL/ADL/', # ADL data
    'flip_net_path': '/Users/kai-chunliu/Documents/code/ssl-wearables-main/model_check_point/mtl_best.mdl', # pre-trained model
    'output_path': 'outputs/',
}

def main(config):
    
    # load label data & ADL data
    df=pd.read_csv(config['load_path']+'Data_table.csv')
    df_r = df.loc[df['hasAG']==1].iloc[:20,:].reset_index(drop=True)
    df_ADL = pd.read_csv(config['load_path_ADL']+'csvVersion(ShiftOnly)_ADL/Data_table.csv')
    
    # create output folder
    t = time.localtime()
    current_time = time.strftime("%m%d%y_%H%M%S_", t)
    config['output_path'] = current_time + config['output_path']
    try:
        os.mkdir(config['output_path'])
    except:
        print(config['output_path']+'is exst')
    Path_Performance_table = config['output_path']
    
    # Model selection
    if config['ModelName'] == 'ResNet':
        # set up hyper-parameter for ResNet
        fc_filterList = [32,64,128,256,512,1024]
        

        Performance_table = np.zeros((config['RunTimes']*len(fc_filterList),12),dtype=np.float32)
        
        indexPerformance = []
        PerformanceInd = 0
        for fc_filter in fc_filterList:
            for run in range(config['RunTimes']):
                # Cross-Validation 
                config['fc_filters'] = fc_filter
                config['output_path'] = Path_Performance_table+f"WS{int(config['window_size']/30)}_FCfilterNum{config['fc_filters']}_Round{run}/"
                indexPerformance.append(f"WS{int(config['window_size']/30)}_FCfilterNum{config['fc_filters']}_Round{run}")
                try:
                    os.mkdir(config['output_path'])
                except:
                    print(config['output_path']+'is exst')
                    
                Performance, Performance_subject_df, confu_AllFold= CrossValidation_SpecifyActivityTrain_ADLTest.LSOCV(df_r, df_ADL, config)
                PerformanceSave(config, Performance, Performance_subject_df, confu_AllFold)
                Performance_table[PerformanceInd,:] = Performance[-1,:]
                PerformanceInd = PerformanceInd+1
            
        

        Performance_table_df=pd.DataFrame(Performance_table, index= indexPerformance, columns=['te_acc','te_sen','te_pre','te_f1', 'tr_acc','tr_sen','tr_pre','tr_f1','vali_acc','vali_sen','vali_pre','vali_f1'])    
        fileName = f"{Path_Performance_table}OverallResults_WindowWise.csv"
        Performance_table_df.to_csv(fileName, index = True, header=True)

    if config['ModelName'] == 'CNNc3f1':
        # set up hyper-parameter for CNN
        filterSizeList = range(31,40,8) #range(31,50,8)
        conv_filterList = [32,64]
        fc_filterList = [16,32]

        Performance_table = np.zeros((len(filterSizeList)*len(conv_filterList)*len(fc_filterList),12),dtype=np.float32)
        indexPerformance = []
        PerformanceInd = 0
        for i in filterSizeList:
            for conv_filter in conv_filterList:
                for fc_filter in fc_filterList:
                    # Cross-Validation 
                    config['filter_width'] = i
                    config['conv_filters'] = conv_filter
                    config['fc_filters'] = fc_filter
                    config['output_path'] = Path_Performance_table+f"WS{int(config['window_size']/30)}_CNNfilterNumber{config['conv_filters']}_CNNfilterWidth{config['filter_width']}_FCfilterNum{config['fc_filters']}/"
                    indexPerformance.append(f"WS{int(config['window_size']/30)}_CNNfilterNumber{config['conv_filters']}_CNNfilterWidth{config['filter_width']}_FCfilterNum{config['fc_filters']}")
                    try:
                        os.mkdir(config['output_path'])
                    except:
                        print(config['output_path']+'is exst')
                
                    Performance, Performance_subject_df, confu_AllFold = CrossValidation_SpecifyActivityTrain_ADLTest.LSOCV(df_r, df_ADL, config)
                    PerformanceSave(config, Performance, Performance_subject_df, confu_AllFold)
                    Performance_table[PerformanceInd,:] = Performance[-1,:]
                    PerformanceInd = PerformanceInd+1
        

        Performance_table_df=pd.DataFrame(Performance_table, index= indexPerformance, columns=['te_acc','te_sen','te_pre','te_f1', 'tr_acc','tr_sen','tr_pre','tr_f1','vali_acc','vali_sen','vali_pre','vali_f1'])    
        fileName = f"{Path_Performance_table}OverallResults_WindowWise.csv"
        Performance_table_df.to_csv(fileName, index = True, header=True)

if __name__ == '__main__':

    config['output_path'] = f"{config['ModelName']}_outputs/"
    main(config)

                
            

