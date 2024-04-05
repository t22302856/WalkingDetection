#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 4 2024

@author: kai-chunliu
"""
import pandas as pd
import numpy as np
import time
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn.utils import class_weight

import torch
from torch.utils.data import DataLoader
from torch import nn
from pytorchtools import EarlyStopping

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, KFold, StratifiedKFold


import data_preprocessing
from utilities import plot_loss, plot2, ConfusionSave
import models
from collections import Counter

import copy


device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")


def load_weights(
    weight_path, model, my_device
):
    # only need to change weights name when
    # the model is trained in a distributed manner

    pretrained_dict = torch.load(weight_path, map_location=my_device)
    pretrained_dict_v2 = copy.deepcopy(
        pretrained_dict
    )  # v2 has the right para names

    # distributed pretraining can be inferred from the keys' module. prefix
    head = next(iter(pretrained_dict_v2)).split('.')[0]  # get head of first key
    if head == 'module':
        # remove module. prefix from dict keys
        pretrained_dict_v2 = {k.partition('module.')[2]: pretrained_dict_v2[k] for k in pretrained_dict_v2.keys()}

    if hasattr(model, 'module'):
        model_dict = model.module.state_dict()
        multi_gpu_ft = True
    else:
        model_dict = model.state_dict()
        multi_gpu_ft = False

    # 1. filter out unnecessary keys such as the final linear layers
    #    we don't want linear layer weights either
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict_v2.items()
        if k in model_dict and k.split(".")[0] != "classifier"
    }

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # 3. load the new state dict
    if multi_gpu_ft:
        model.module.load_state_dict(model_dict)
    else:
        model.load_state_dict(model_dict)
    print("%d Weights loaded" % len(pretrained_dict))

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm1d") != -1:
        m.eval()
        
def freeze_weights(model):
    i = 0
    # Set Batch_norm running stats to be frozen
    # Only freezing ConV layers for now
    # or it will lead to bad results
    # http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/
    for name, param in model.named_parameters():
        if name.split(".")[0] == "feature_extractor":
            param.requires_grad = False
            i += 1
    print("Weights being frozen: %d" % i)
    model.apply(set_bn_eval)

def init_model(config, device):
    if config['resnet_version'] > 0:
        model = models.Resnet(
            output_size=config['nb_classes'],
            is_eva=True,
            resnet_version=1,
            epoch_len=10,
            config=config
        )

    model.to(device, dtype=torch.float)
    return model

def setup_model(config, my_device):
    model = init_model(config, device)

    if config['load_weights']:
        load_weights(
            config['flip_net_path'],
            model,
            my_device
        )
    if config['freeze_weight']:
        freeze_weights(model)
    return model



def train(model, trainloader, optimizer, criterion):
    # helper objects needed for proper documentation
    train_epoch_losses = []
    train_epoch_preds = []
    train_epoch_gt = []
    start_time = time.time()

    # iterate over the trainloader object (it'll return batches which you can use)
    model.train()
    for i, (x, y) in enumerate(trainloader):
        # sends batch x and y to the GPU
        if x.size()[0]>1:
            inputs, targets = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # send inputs through network to get predictions
            train_output = model(inputs)
                        
            
            # calculates loss
            loss = criterion(train_output, targets.long())
            
            # backprogate your computed loss through the network
            # use the .backward() and .step() function on your loss and optimizer
            loss.backward()
            optimizer.step()
            
            # calculate actual predictions (i.e. softmax probabilites); use torch.nn.functional.softmax()
            train_output = torch.nn.functional.softmax(train_output, dim=1)
            
            # appends the computed batch loss to list
            train_epoch_losses.append(loss.item())
            
            
            # creates predictions and true labels; appends them to the final lists
            y_preds = np.argmax(train_output.cpu().detach().numpy(), axis=-1)
            y_true = targets.cpu().numpy().flatten()
            train_epoch_preds = np.concatenate((np.array(train_epoch_preds, int), np.array(y_preds, int)))
            train_epoch_gt = np.concatenate((np.array(train_epoch_gt, int), np.array(y_true, int)))

        
        
    elapsed = time.time() - start_time
    return train_epoch_losses, train_epoch_preds,train_epoch_gt, elapsed 

def validate(model, testloader, criterion):
    # helper objects
    test_epoch_preds = []
    test_epoch_gt = []
    test_epoch_losses = []

    # sets network to eval mode and 
    model.eval()
    with torch.no_grad():
    # iterate over the valloader object (it'll return batches which you can use)
        for i, (x, y) in enumerate(testloader):
            # sends batch x and y to the GPU
            inputs, targets = x.to(device), y.to(device)

            # send inputs through network to get predictions
            test_output = model(inputs)

            # calculates loss by passing criterion both predictions and true labels 
            test_loss = criterion(test_output, targets.long())

            # calculate actual predictions (i.e. softmax probabilites); use torch.nn.functional.softmax() on dim=1
            test_output = torch.nn.functional.softmax(test_output, dim=1)

            # appends test loss to list
            test_epoch_losses.append(test_loss.item())

            # creates predictions and true labels; appends them to the final lists
            y_preds = np.argmax(test_output.cpu().numpy(), axis=-1)
            y_true = targets.cpu().numpy().flatten()
            test_epoch_preds = np.concatenate((np.array(test_epoch_preds, int), np.array(y_preds, int)))
            test_epoch_gt = np.concatenate((np.array(test_epoch_gt, int), np.array(y_true, int)))
        return test_epoch_losses, test_epoch_preds,test_epoch_gt

def test_window(Performance_subject, Performance, Performance_ind, SubjIDList_test, testloader, model, criterion, config):
    # helper objects
    test_epoch_preds = []
    test_epoch_gt = []


    model.eval()
    with torch.no_grad():
    # iterate over the valloader object (it'll return batches which you can use)
        for i, (x, y) in enumerate(testloader):
            # sends batch x and y to the GPU
            inputs, targets = x.to(device), y.to(device)

            # send inputs through network to get predictions
            test_output = model(inputs)

            # calculate actual predictions (i.e. softmax probabilites); use torch.nn.functional.softmax() on dim=1
            test_output = torch.nn.functional.softmax(test_output, dim=1)


            # creates predictions and true labels; appends them to the final lists
            y_preds = np.argmax(test_output.cpu().numpy(), axis=-1)
            y_true = targets.cpu().numpy().flatten()
            test_epoch_preds = np.concatenate((np.array(test_epoch_preds, int), np.array(y_preds, int)))
            test_epoch_gt = np.concatenate((np.array(test_epoch_gt, int), np.array(y_true, int)))

    
    subject_test = np.unique(SubjIDList_test)
    for s in range(len(subject_test)):
        c_subject = subject_test[s]
        c_ind = np.where(SubjIDList_test==c_subject)
        c_test_epoch_gt = test_epoch_gt[c_ind]
        c_test_epoch_preds = test_epoch_preds[c_ind]
        # classification_report(c_test_epoch_gt, c_test_epoch_preds)
        subpath = 'ConfusionSubject'
        fileName = f"ConfusionSubject_{c_subject}.csv"
        confu_subject = ConfusionSave(config,c_test_epoch_gt, c_test_epoch_preds, subpath, fileName)
        
        tn, fp, fn, tp = confusion_matrix(c_test_epoch_gt, c_test_epoch_preds,labels=np.array([0,1])).ravel()
        Acc= accuracy_score(c_test_epoch_gt, c_test_epoch_preds)
        Sen= recall_score(c_test_epoch_gt, c_test_epoch_preds, labels=np.unique(c_test_epoch_gt), average=None)[1]
        Pre= precision_score(c_test_epoch_gt, c_test_epoch_preds, average=None, labels=np.unique(c_test_epoch_gt))[1]
        f1= f1_score(c_test_epoch_gt, c_test_epoch_preds, average=None, labels=np.unique(c_test_epoch_gt))[1]
        Performance_subject.loc[c_subject] = np.array([Performance_ind, Acc, Sen, Pre, f1])
    
    # classification_report(test_epoch_gt, test_epoch_preds)
    # save data
    subpath = 'ConfusionFold'
    fileName = f"ConfusionFold_{config['fold']}fold{config['cur_fold']}.csv"
    confu = ConfusionSave(config,test_epoch_gt, test_epoch_preds, subpath, fileName)
 
    
    
    Acc= accuracy_score(test_epoch_gt, test_epoch_preds)
    Sen= recall_score(test_epoch_gt, test_epoch_preds, average=None, labels=np.unique(test_epoch_gt))[1]
    Pre= precision_score(test_epoch_gt, test_epoch_preds, average=None, labels=np.unique(test_epoch_gt))[1]
    f1= f1_score(test_epoch_gt, test_epoch_preds, average=None, labels=np.unique(test_epoch_gt))[1]
    Performance[Performance_ind,0:4]=np.array([Acc, Sen, Pre, f1])                
    
    
    return Performance, Performance_subject, confu 

def LSOCV(df_r, df_ADL, config): #leave subjects out cross validation

    #list the unique subject index
    subject_ind=df_ADL["subjID"].unique()
    gkf = KFold(n_splits = config["fold"],shuffle=True, random_state=config['seed'])

    #split data into k-folds
    kfold_list =  []
    kfoldInd_list = []
    for i, (train_index, test_index) in enumerate(gkf.split(subject_ind, subject_ind, subject_ind)):
        c_subject = subject_ind[test_index]
        c_subject_ind = df_ADL[df_ADL['subjID'].isin(c_subject)].index.tolist()
        kfold_list.append(subject_ind[test_index])
        kfoldInd_list.append(c_subject_ind)

    #prepare performance tables for k-fold
    #allPerformance
    Performance_subject = np.zeros((len(subject_ind),5),dtype=np.float32)
    Performance_subject = pd.DataFrame(Performance_subject, 
                                       columns=['fold','test_acc','test_sen','test_pre','test_f1'],
                                       index = subject_ind)
    
    # 0-3: testing Acc, Sen, Pre, f1, 4-7: training Acc, Sen, Pre, f1, 8-11: Vlidation Acc, Sen, Pre, f1
    Performance=np.zeros((config['fold'],12),dtype=np.float32)
    confu_AllFold = np.zeros((config['nb_classes'], config['nb_classes']),dtype=np.int32 )
    
    # starting k-fold cross-validation
    Performance_ind=0    
    for Performance_ind in range(len(kfold_list)):
        ## windowing & labeling ##
        #ADL data preparation
        config['cur_fold']= Performance_ind    
        df_ADL_test = df_ADL[df_ADL['subjID'].isin(kfold_list[Performance_ind])]
        df_ADL_vali  = df_ADL[df_ADL['subjID'].isin(kfold_list[(Performance_ind+1)%config['fold']])]
        df_ADL_train = df_ADL[~df_ADL['subjID'].isin(kfold_list[(Performance_ind+1)%config['fold']]) & ~df_ADL['subjID'].isin(kfold_list[Performance_ind])]
        x_ADL_train, y_ADL_train, SubjIDList_train = data_preprocessing.create_segments_and_labels_ADL(config,df_ADL_train)
        x_ADL_vali, y_ADL_vali, SubjIDList_vali = data_preprocessing.create_segments_and_labels_ADL(config,df_ADL_vali)
        x_ADL_test, y_ADL_test, SubjIDList_test = data_preprocessing.create_segments_and_labels_ADL(config,df_ADL_test)
        
        #Specified Activity data preparation
        if config['SpecifiedActivity_train']:
            #list the unique subject index
            subject_r_ind=df_r["subjID"].unique()
            gkf_r = KFold(n_splits =config["fold"],shuffle=True, random_state=config['seed'])

            #split group
            kfold_r_list =  []
            kfoldInd_r_list = []
            for i, (train_index, test_index) in enumerate(gkf_r.split(subject_r_ind, subject_r_ind, subject_r_ind)):
                c_subject_r = subject_r_ind[test_index]
                c_subject_r_ind = df_r[df_r['subjID'].isin(c_subject_r)].index.tolist()
                kfold_r_list.append(subject_r_ind[test_index])
                kfoldInd_r_list.append(c_subject_r_ind)
            
            config['cur_fold']= Performance_ind
            df_r_vali = df_r[df_r['subjID'].isin(kfold_r_list[Performance_ind])]            
            df_r_train = df_r[~df_r['subjID'].isin(kfold_list[Performance_ind])]
            
            # y_r_train1: activity label, y_r_train2: binary label for walking(1) & non-walking (0)
            x_r_train, y_r_train1, y_r_train2= data_preprocessing.create_segments_and_labels(config,df_r_train)
            x_r_vali, y_r_vali1, y_r_vali2 = data_preprocessing.create_segments_and_labels(config,df_r_vali)
        
        
        
        if config['SpecifiedActivity_train'] & config['ADL_train']:
            # mixed training
            x_train = np.concatenate((x_r_train,x_ADL_train))
            x_vali = np.concatenate((x_r_vali,x_ADL_vali))
            x_test = x_ADL_test
            y_train = np.concatenate((y_r_train2,y_ADL_train))
            y_vali = np.concatenate((y_r_vali2,y_ADL_vali))
            y_test = y_ADL_test
        elif config['SpecifiedActivity_train'] & ~config['ADL_train']: 
            # labeled data training only
            x_train = x_r_train
            x_vali = x_r_vali
            x_test = x_ADL_test
            y_train = y_r_train2
            y_vali = y_r_vali2
            y_test = y_ADL_test
        elif ~config['SpecifiedActivity_train'] & config['ADL_train']: 
            # ADL data training only
            x_train = x_ADL_train
            x_vali = x_ADL_vali
            x_test = x_ADL_test
            y_train = y_ADL_train
            y_vali = y_ADL_vali
            y_test = y_ADL_test
        else:
            print('error: no training and testing data')
            break
        
        if config['ModelName'] == 'ResNet':
            x_train = x_train.reshape((-1,config['nb_channels'],config['window_size']))
            x_vali = x_vali.reshape((-1,config['nb_channels'],config['window_size']))
            x_test = x_test.reshape((-1,config['nb_channels'],config['window_size']))
        
        ## prepareing for pytorch format ##
        x_train, x_vali, x_test = x_train.astype('float32'), x_vali.astype('float32'), x_test.astype('float32')
        config['nb_classes'] = len(np.unique(y_train))
        
        # Packing data in torch format
        train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float().to(device), torch.from_numpy(y_train).to(device))
        vali_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_vali).float().to(device), torch.from_numpy(y_vali).to(device))
        test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test).float().to(device), torch.from_numpy(y_test).to(device))
        
        trainloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)     
        valiloader = DataLoader(vali_dataset, batch_size=config['batch_size'], shuffle=False)     
        testloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        #impot model
        # network = DeepConvLSTM(config)
        if config['ModelName'] == 'CNNc3f1':
            model = models.CNNc3f1(config).to(device)
        elif config['ModelName'] == 'ResNet':
            model = setup_model(config, device)
        
        # initialize the optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        if config['LoosFunction'] == 'weight':
            class_weights=class_weight.compute_class_weight('balanced',classes = np.unique(y_train),y = y_train)
            class_weights=torch.tensor(class_weights,dtype=torch.float)
            criterion = nn.CrossEntropyLoss(weight=class_weights,reduction='mean').to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)
        

        
        # initialize the early_stopping object
        if config['early_stopping']:
            early_stopping = EarlyStopping(patience=config['patience'], verbose=True)
        
        loss_train, loss_vali, loss_test  = [], [], []
        acc_train, acc_vali,acc_test = [], [], []
        f1_train, f1_vali, f1_test = [], [], []
        
        
        # define your training loop; iterates over the number of epochs
        for e in range(config['epochs']):
            
            train_epoch_losses, train_epoch_preds,train_epoch_gt, elapsed = train(model, trainloader, optimizer, criterion)
            vali_epoch_losses, vali_epoch_preds,vali_epoch_gt = validate(model, valiloader, criterion)
            test_epoch_losses, test_epoch_preds,test_epoch_gt = validate(model, testloader, criterion)
            
            loss_train.append(np.mean(train_epoch_losses))
            loss_vali.append(np.mean(vali_epoch_losses))
            loss_test.append(np.mean(test_epoch_losses))
            acc_train.append(accuracy_score(train_epoch_gt, train_epoch_preds))
            acc_vali.append(accuracy_score(vali_epoch_gt, vali_epoch_preds))
            acc_test.append(accuracy_score(test_epoch_gt, test_epoch_preds))
            f1_train.append(f1_score(train_epoch_gt, train_epoch_preds, average='macro'))
            f1_vali.append(f1_score(vali_epoch_gt, vali_epoch_preds, average='macro'))
            f1_test.append(f1_score(test_epoch_gt, test_epoch_preds, average='macro'))
            
            
            print("\nEPOCH: {}/{}".format(e + 1, config['epochs']),
                  "\n {:5.4f} s/epoch".format(elapsed),
                  "\nTrain Loss: {:.4f}".format(np.mean(train_epoch_losses)),
                  "Train Acc: {:.4f}".format(accuracy_score(train_epoch_gt, train_epoch_preds)),
                  # "Train Prec: {:.4f}".format(precision_score(train_epoch_gt, train_epoch_preds, average='macro')),
                  # "Train Rcll: {:.4f}".format(recall_score(train_epoch_gt, train_epoch_preds, average='macro')),
                   "Train F1: {:.4f}".format(f1_score(train_epoch_gt, train_epoch_preds, average=None)[1]),
                  "\nVali Loss: {:.4f}".format(np.mean(vali_epoch_losses)),
                  "Vali Acc: {:.4f}".format(accuracy_score(vali_epoch_gt, vali_epoch_preds)),
                  # "Vali Prec: {:.4f}".format(precision_score(vali_epoch_gt, vali_epoch_preds, average='macro')),
                  # "Vali Rcll: {:.4f}".format(recall_score(vali_epoch_gt, vali_epoch_preds, average='macro')),
                   "Vali F1: {:.4f}".format(f1_score(vali_epoch_gt, vali_epoch_preds, average=None)[1]),
                  "\nTest Loss: {:.4f}".format(np.mean(test_epoch_losses)),
                  "Test Acc: {:.4f}".format(accuracy_score(test_epoch_gt, test_epoch_preds)),
                  "Test F1: {:.4f}".format(f1_score(test_epoch_gt, test_epoch_preds, average=None)[1]))
            
            # save_best_model(np.mean(vali_epoch_losses), e, model, optimizer, criterion)
            # print('-'*50)
            if config['early_stopping']:
                early_stopping(np.mean(vali_epoch_losses), model) 
                # early_stopping(f1_score(vali_epoch_gt, vali_epoch_preds, average='macro')*-1, model) 
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                    
            
        # load the last checkpoint with the best model
        if config['early_stopping']:
            model.load_state_dict(torch.load('checkpoint.pt'))
        
        #plot classification results
        if config['Plot_Flag']:
            plot2(loss_train, loss_vali, loss_test, acc_train, acc_vali, acc_test, f1_train, f1_vali, f1_test, config)
        
        #-------Recording testing, training, and validation reuslts based on windows------
        #testing
        Performance, Performance_subject_df, confu_fold  = test_window(Performance_subject, Performance, Performance_ind, SubjIDList_test, testloader, model, criterion, config)
        confu_AllFold = confu_AllFold + confu_fold
        
        
        #training
        train_epoch_losses, train_epoch_preds,train_epoch_gt = validate(model, trainloader, criterion)
        Acc= accuracy_score(train_epoch_gt, train_epoch_preds)
        Sen= recall_score(train_epoch_gt, train_epoch_preds, average=None, labels=np.unique(test_epoch_gt))[1]
        Pre= precision_score(train_epoch_gt, train_epoch_preds, average=None, labels=np.unique(test_epoch_gt))[1]
        f1= f1_score(train_epoch_gt, train_epoch_preds, average=None, labels=np.unique(test_epoch_gt))[1]
        Performance[Performance_ind,4:8]=np.array([Acc, Sen, Pre, f1])  
        
        #validation
        vali_epoch_losses, vali_epoch_preds,vali_epoch_gt = validate(model, valiloader, criterion)
        Acc= accuracy_score(vali_epoch_gt, vali_epoch_preds)
        Sen= recall_score(vali_epoch_gt, vali_epoch_preds, average=None, labels=np.unique(vali_epoch_gt))[1]
        Pre= precision_score(vali_epoch_gt, vali_epoch_preds, average=None, labels=np.unique(vali_epoch_gt))[1]
        f1= f1_score(vali_epoch_gt, vali_epoch_preds, average=None, labels=np.unique(vali_epoch_gt))[1]
        Performance[Performance_ind,8:]=np.array([Acc, Sen, Pre, f1])

        
        
    #overal performance concate
    Performance = np.concatenate((Performance, Performance.mean(axis=0).reshape(1,-1)),axis=0)
    
    return Performance, Performance_subject_df, confu_AllFold




