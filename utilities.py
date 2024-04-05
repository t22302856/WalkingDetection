#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 4 2024

@author: kai-chunliu
"""

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches as pc
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

import numpy as np
import os
from scipy import stats
import pandas as pd

import torch
from torch.utils.data import DataLoader


def signalPlot(c_data, label, subject,n=0):

    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(c_data)+1),c_data[:,0], label='x-axis Acc(m/s^2)')
    plt.plot(range(1,len(c_data)+1),c_data[:,1], label='y-axis Acc(m/s^2)')
    plt.plot(range(1,len(c_data)+1),c_data[:,2], label='z-axis Acc(m/s^2)')
    plt.xlabel('Data Point')
    # plt.grid(True)
    plt.legend(loc='upper left')
    plt.ylim(-3,3)
    plt.title(label+'_S'+subject+'_W'+str(n), fontsize = 20)
    plt.show()

def PerformanceSave(config, Performance, Performance_subject_df, confu_AllFold):
    indexPerformance = [str(i) for i in range(config['fold'])]
    indexPerformance.append('overall')

    Performance_df=pd.DataFrame(Performance, index= indexPerformance, columns=['te_acc','te_sen','te_pre','te_f1', 'tr_acc','tr_sen','tr_pre','tr_f1','vali_acc','vali_sen','vali_pre','vali_f1'])    
    fileName = f"{config['output_path']}WindowResults_WS{int(config['window_size']/config['FS'])}_CNNfilterNumber{config['conv_filters']}_CNNfilterWidth{config['filter_width']}_FCfilterNum{config['fc_filters']}.csv"
    Performance_df.to_csv(fileName, index = True, header=True)

    fileName = f"{config['output_path']}WindowResultsSubject_WS{int(config['window_size']/config['FS'])}_CNNfilterNumber{config['conv_filters']}_CNNfilterWidth{config['filter_width']}_FCfilterNum{config['fc_filters']}.csv"
    Performance_subject_df.to_csv(fileName, index = True, header=True)


    confu_AllFold_df = pd.DataFrame(confu_AllFold, index=['Walking', 'Non-Walking'], columns=['Walking', 'Non-Walking'])        
    fileName = f"{config['output_path']}confu_AllFold_WS{int(config['window_size']/30)}_CNNfilterNumber{config['conv_filters']}_CNNfilterWidth{config['filter_width']}_FCfilterNum{config['fc_filters']}.csv"
    confu_AllFold_df.to_csv(fileName, index = True, header=True)

    
    
def ConfusionSave(config,GT,pred,subpath, fileName):
       
    confu = confusion_matrix(GT,pred, labels=[1,0])
    confu_df = pd.DataFrame(confu, index=['Walking', 'Non-Walking'], columns=['Walking', 'Non-Walking'])
    try:
        os.mkdir(config['output_path']+subpath)
    except:
        print(config['output_path']+subpath+'is exst')
    SaveName = f"{config['output_path']}/{subpath}/{fileName}"
    confu_df.to_csv(SaveName, index = True, header=True)
    
    return confu
    

def plot2(train_loss, valid_loss, test_loss, acc_train, acc_vali, acc_test, f1_train, f1_vali, f1_test, config):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')
    plt.plot(range(1,len(valid_loss)+1),test_loss,label='Test Loss')
    # plt.plot(range(1,len(acc_train)+1),acc_train, label='Training Acc')
    # plt.plot(range(1,len(acc_vali)+1),acc_vali, label='Validation Acc')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    # plt.yscale("log")
    # plt.ylim(0, 1) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    try:
        os.mkdir(config['output_path']+'/LossCurve/')
    except:
        print(config['output_path']+'/LossCurve/'+'is exst')
    fileName = f"{config['output_path']}/LossCurve/LossPlot_{config['fold']}fold{config['cur_fold']}.png"
    fig.savefig(fileName, bbox_inches='tight')

    # visualize the Accuracy as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(acc_train)+1),acc_train, label='Training Acc')
    plt.plot(range(1,len(acc_vali)+1),acc_vali, label='Validation Acc')
    plt.plot(range(1,len(acc_test)+1),acc_test, label='Testing Acc')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    # plt.ylabel('loss')
    plt.ylim(0, 1) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    
    fileName = f"{config['output_path']}/LossCurve/AccPlot_{config['fold']}fold{config['cur_fold']}.png"
    fig.savefig(fileName, bbox_inches='tight')
    
    # visualize the f1-scroe as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(f1_train)+1),f1_train, label='Training f1')
    plt.plot(range(1,len(f1_vali)+1),f1_vali, label='Validation f1')
    plt.plot(range(1,len(f1_test)+1),f1_test, label='Testing f1')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    # plt.ylabel('loss')
    plt.ylim(0, 1) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    
    fileName = f"{config['output_path']}/LossCurve/f1Plot_{config['fold']}fold{config['cur_fold']}.png"
    fig.savefig(fileName, bbox_inches='tight')


def plot_loss(train_losses,test_losses, fold, floag,config):
    plt.figure(figsize=(10,5))
    plt.title("Loss Curve")
    plt.plot(train_losses,label="train")
    plt.plot(test_losses,label="test")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig('books_read.png')

        
def show_confusion_matrix(validations, predictions, LABELS,config):

    matrix = confusion_matrix(validations, predictions)
    fig = plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    fileName = f"{config['output_path']}Confusion_WS{int(config['window_size']/30)}_CNNfilterNumber{config['conv_filters']}_CNNfilterWidth{config['filter_width']}_FCfilterNum{config['fc_filters']}.png"
    fig.savefig(fileName, bbox_inches='tight')
   
    
        
    