#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:24:35 2023

@author: kai-chunliu
"""
import torch
import torch.nn as nn
from math import floor
import numpy as np
import torch.nn.functional as F

class CNN2Dc2f1(nn.Module):
    def __init__(self, config):
        super(CNN2Dc2f1, self).__init__()
        # parameters
        self.window_size = config['window_size']
        self.drop_prob = config['drop_prob']
        self.nb_channels = config['nb_channels']
        self.nb_classes = config['nb_classes']
        self.seed = config['seed']
        self.conv_filters = config['conv_filters']
        self.fc_filters = config['fc_filters']
        self.filter_width = config['filter_width']
        self.max2d_width = config['max1d_width']


        # define activation function
        self.relu = nn.ReLU(inplace=True)

        # define conv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.conv_filters, [self.filter_width,1]),
            nn.BatchNorm2d(self.conv_filters),
            nn.ReLU(),
            nn.MaxPool2d([self.max2d_width,1], stride=1),
            nn.Dropout(p=self.drop_prob)
            )
        Output1_high = (self.window_size-self.filter_width+1)-self.max2d_width+1
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.conv_filters, self.conv_filters, [self.filter_width,1]),
            nn.BatchNorm2d(self.conv_filters),
            nn.ReLU(),
            nn.MaxPool2d([self.max2d_width,1], stride=1),
            nn.Dropout(p=self.drop_prob)
            )
        Output2_high = (Output1_high-self.filter_width+1)-self.max2d_width+1
        
        self.fc1 = nn.Sequential(
            nn.Linear((Output2_high)*self.conv_filters*self.nb_channels, self.fc_filters),
            nn.ReLU(),
            nn.Linear(self.fc_filters, self.nb_classes)
            )
        
        

    def forward(self, x):
        # reshape data for convolutions
        x = x.view(-1, 1, self.window_size, self.nb_channels)
        
        # apply convolution and the activation function
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        # out = self.dropout(out)
        out = self.fc1(out)

        return out

class CNN2Dc3f1(nn.Module):
    def __init__(self, config):
        super(CNN2Dc3f1, self).__init__()
        # parameters
        self.window_size = config['window_size']
        self.drop_prob = config['drop_prob']
        self.nb_channels = config['nb_channels']
        self.nb_classes = config['nb_classes']
        self.seed = config['seed']
        self.conv_filters = config['conv_filters']
        self.fc_filters = config['fc_filters']
        self.filter_width = config['filter_width']
        self.max2d_width = config['max1d_width']


        # define activation function
        self.relu = nn.ReLU(inplace=True)

        # define conv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.conv_filters, [self.filter_width,1]),
            nn.BatchNorm2d(self.conv_filters),
            nn.ReLU(),
            nn.MaxPool2d([self.max2d_width,1], stride=1),
            nn.Dropout(p=self.drop_prob)
            )
        Output1_high = (self.window_size-self.filter_width+1)-self.max2d_width+1
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.conv_filters, self.conv_filters, [self.filter_width,1]),
            nn.BatchNorm2d(self.conv_filters),
            nn.ReLU(),
            nn.MaxPool2d([self.max2d_width,1], stride=1),
            nn.Dropout(p=self.drop_prob)
            )
        Output2_high = (Output1_high-self.filter_width+1)-self.max2d_width+1
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.conv_filters, self.conv_filters, [self.filter_width,1]),
            nn.BatchNorm2d(self.conv_filters),
            nn.ReLU(),
            nn.MaxPool2d([self.max2d_width,1], stride=1),
            nn.Dropout(p=self.drop_prob)
            )
        Output3_high = (Output2_high-self.filter_width+1)-self.max2d_width+1
        
        self.fc1 = nn.Sequential(
            nn.Linear((Output3_high)*self.conv_filters*self.nb_channels, self.fc_filters),
            nn.ReLU(),
            nn.Linear(self.fc_filters, self.nb_classes)
            )
        
        

    def forward(self, x):
        # reshape data for convolutions
        x = x.view(-1, 1, self.window_size, self.nb_channels)
        
        # apply convolution and the activation function
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        # out = self.dropout(out)
        out = self.fc1(out)

        return out

class CNN2Dc4f1(nn.Module):
    def __init__(self, config):
        super(CNN2Dc4f1, self).__init__()
        # parameters
        self.window_size = config['window_size']
        self.drop_prob = config['drop_prob']
        self.nb_channels = config['nb_channels']
        self.nb_classes = config['nb_classes']
        self.seed = config['seed']
        self.conv_filters = config['conv_filters']
        self.fc_filters = config['fc_filters']
        self.filter_width = config['filter_width']
        self.max2d_width = config['max1d_width']


        # define activation function
        self.relu = nn.ReLU(inplace=True)

        # define conv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.conv_filters, [self.filter_width,1]),
            nn.BatchNorm2d(self.conv_filters),
            nn.ReLU(),
            nn.MaxPool2d([self.max2d_width,1], stride=1),
            nn.Dropout(p=self.drop_prob)
            )
        Output1_high = (self.window_size-self.filter_width+1)-self.max2d_width+1
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.conv_filters, self.conv_filters, [self.filter_width,1]),
            nn.BatchNorm2d(self.conv_filters),
            nn.ReLU(),
            nn.MaxPool2d([self.max2d_width,1], stride=1),
            nn.Dropout(p=self.drop_prob)
            )
        Output2_high = (Output1_high-self.filter_width+1)-self.max2d_width+1
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.conv_filters, self.conv_filters, [self.filter_width,1]),
            nn.BatchNorm2d(self.conv_filters),
            nn.ReLU(),
            nn.MaxPool2d([self.max2d_width,1], stride=1),
            nn.Dropout(p=self.drop_prob)
            )
        Output3_high = (Output2_high-self.filter_width+1)-self.max2d_width+1
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.conv_filters, self.conv_filters, [self.filter_width,1]),
            nn.BatchNorm2d(self.conv_filters),
            nn.ReLU(),
            nn.MaxPool2d([self.max2d_width,1], stride=1),
            nn.Dropout(p=self.drop_prob)
            )
        Output4_high = (Output3_high-self.filter_width+1)-self.max2d_width+1
        
        
        self.fc1 = nn.Sequential(
            nn.Linear((Output4_high)*self.conv_filters*self.nb_channels, self.fc_filters),
            nn.ReLU(),
            nn.Linear(self.fc_filters, self.nb_classes)
            )
        
        

    def forward(self, x):
        # reshape data for convolutions
        x = x.view(-1, 1, self.window_size, self.nb_channels)
        
        # apply convolution and the activation function
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.size(0), -1)
        # out = self.dropout(out)
        out = self.fc1(out)

        return out     

class CNNc2f1(nn.Module):
    def __init__(self, config):
        super(CNNc2f1, self).__init__()
        # parameters
        self.window_size = config['window_size']
        self.drop_prob = config['drop_prob']
        self.nb_channels = config['nb_channels']
        self.nb_classes = config['nb_classes']
        self.seed = config['seed']
        self.conv_filters = config['conv_filters']
        self.fc_filters = config['fc_filters']
        self.filter_width = config['filter_width']
        self.max1d_width = config['max1d_width']


        # define activation function
        self.relu = nn.ReLU(inplace=True)

        # define conv layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.nb_channels, self.conv_filters, 3, stride=1, padding =1),
            nn.BatchNorm1d(self.conv_filters),
            nn.ReLU(),
            nn.Dropout(p=self.drop_prob)
            )
        Output1_high = self.window_size
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.conv_filters, self.conv_filters*2, self.filter_width, stride=2),
            nn.BatchNorm1d(self.conv_filters*2),
            nn.ReLU(),
            nn.Dropout(p=self.drop_prob)
            )
        Output2_high = floor((Output1_high-self.filter_width)/2)+1
        
        self.fc1 = nn.Sequential(
            nn.Linear((Output2_high)*self.conv_filters*2, self.fc_filters),
            nn.ReLU(),
            nn.Linear(self.fc_filters, self.nb_classes)
            )
        
        

    def forward(self, x):
        # reshape data for convolutions
        x = x.view(-1, self.nb_channels, self.window_size)
        
        # apply convolution and the activation function
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        # out = self.dropout(out)
        out = self.fc1(out)

        return out

class CNNc3f1(nn.Module):
    def __init__(self, config):
        super(CNNc3f1, self).__init__()
        # parameters
        self.window_size = config['window_size']
        self.drop_prob = config['drop_prob']
        self.nb_channels = config['nb_channels']
        self.nb_classes = config['nb_classes']
        self.seed = config['seed']
        self.conv_filters = config['conv_filters']
        self.fc_filters = config['fc_filters']
        self.filter_width = config['filter_width']
        self.max1d_width = config['max1d_width']


        # define activation function
        self.relu = nn.ReLU(inplace=True)

        # define conv layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.nb_channels, self.conv_filters, 3, stride=1, padding =1),
            nn.BatchNorm1d(self.conv_filters),
            nn.ReLU(),
            # nn.Dropout(p=self.drop_prob)
            )
        Output1_high = self.window_size
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.conv_filters, self.conv_filters*2, self.filter_width, stride=2),
            nn.BatchNorm1d(self.conv_filters*2),
            nn.ReLU(),
            # nn.Dropout(p=self.drop_prob)
            )
        Output2_high = floor((Output1_high-self.filter_width)/2)+1
        self.conv3 = nn.Sequential(
            nn.Conv1d(self.conv_filters*2, self.conv_filters*2*2, self.filter_width, stride=2),
            nn.BatchNorm1d(self.conv_filters*2*2),
            nn.ReLU(),
            # nn.Dropout(p=self.drop_prob)
            )
        Output3_high = floor((Output2_high-self.filter_width)/2)+1
        
        self.fc1 = nn.Sequential(
            nn.Linear((Output3_high)*self.conv_filters*2*2, self.fc_filters),
            nn.BatchNorm1d(self.fc_filters),
            nn.ReLU(),
            )
        self.fc2 = nn.Linear(self.fc_filters, self.nb_classes)
        
        

    def forward(self, x):
        # reshape data for convolutions
        x = x.view(-1, self.nb_channels, self.window_size)
        
        # apply convolution and the activation function
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        # out = self.dropout(out)z
        out = self.fc1(out)
        out = self.fc2(out)

        return out

class CNNc4f1(nn.Module):
    def __init__(self, config):
        super(CNNc4f1, self).__init__()
        # parameters
        self.window_size = config['window_size']
        self.drop_prob = config['drop_prob']
        self.nb_channels = config['nb_channels']
        self.nb_classes = config['nb_classes']
        self.seed = config['seed']
        self.conv_filters = config['conv_filters']
        self.fc_filters = config['fc_filters']
        self.filter_width = config['filter_width']
        self.max1d_width = config['max1d_width']


        # define activation function
        self.relu = nn.ReLU(inplace=True)

        # define conv layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.nb_channels, self.conv_filters, 3, stride=1, padding =1),
            nn.BatchNorm1d(self.conv_filters),
            nn.ReLU(),
            # nn.Dropout(p=self.drop_prob)
            )
        Output1_high = self.window_size
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.conv_filters, self.conv_filters*2, self.filter_width, stride=2),
            nn.BatchNorm1d(self.conv_filters*2),
            nn.ReLU(),
            # nn.Dropout(p=self.drop_prob)
            )
        Output2_high = floor((Output1_high-self.filter_width)/2)+1
        self.conv3 = nn.Sequential(
            nn.Conv1d(self.conv_filters*2, self.conv_filters*2*2, self.filter_width, stride=2),
            nn.BatchNorm1d(self.conv_filters*2*2),
            nn.ReLU(),
            # nn.Dropout(p=self.drop_prob)
            )
        Output3_high = floor((Output2_high-self.filter_width)/2)+1
        self.conv4 = nn.Sequential(
            nn.Conv1d(self.conv_filters*2*2, self.conv_filters*2*2*2, self.filter_width, stride=2),
            nn.BatchNorm1d(self.conv_filters*2*2*2),
            nn.ReLU(),
            # nn.Dropout(p=self.drop_prob)
            )
        Output4_high = floor((Output3_high-self.filter_width)/2)+1
        
        self.fc1 = nn.Sequential(
            nn.Linear((Output4_high)*self.conv_filters*2*2*2, self.fc_filters),
            nn.BatchNorm1d(self.fc_filters),
            nn.ReLU(),
            nn.Linear(self.fc_filters, self.nb_classes)
            )
        self.fc_output = nn.Linear((Output4_high)*self.conv_filters*2*2*2, self.nb_classes)
        
        

    def forward(self, x):
        # reshape data for convolutions
        x = x.view(-1, self.nb_channels, self.window_size)
        
        # apply convolution and the activation function
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.size(0), -1)
        # out = self.dropout(out)
        out = self.fc1(out)

        return out

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         # parameters

#         # define activation function
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv1d(inplanes, planes, 3,stride, padding=1,bias=False)
#         self.bn1 = nn.BatchNorm1d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv1d(planes, planes, 3,stride=1, padding=1,bias=False)
#         self.bn2 = nn.BatchNorm1d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):

#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out    


# class ResNet3(nn.Module):
#     def __init__(self, config, block, layers, num_classes = 2):
#         super(ResNet3, self).__init__()
#         self.nb_channels = config['nb_channels']
#         self.window_size = config['window_size']
#         num_classes = config['nb_classes']       
#         self.inplanes = config['conv_filters']
#         self.conv1 = nn.Sequential(
#                         nn.Conv1d(config['nb_channels'], config['conv_filters'], kernel_size = 7, stride = 2, padding = 3),
#                         nn.BatchNorm1d(config['conv_filters']),
#                         nn.ReLU())
#         self.maxpool = nn.MaxPool1d(kernel_size = 3, stride = 2, padding = 1)
#         self.layer0 = self._make_layer(config['ResNetBlock'], block, config['conv_filters'], layers[0], stride = 1)
#         self.layer1 = self._make_layer(config['ResNetBlock'], block, config['conv_filters']*2, layers[1], stride = 2)
#         self.layer2 = self._make_layer(config['ResNetBlock'], block, config['conv_filters']*2*2, layers[2], stride = 2)
#         # self.layer3 = self._make_layer(config['ResNetBlock'], block, config['conv_filters']*2*2*2, layers[2], stride = 2)
#         self.avgpool = nn.AvgPool1d(19, stride=1)
#         self.fc = nn.Linear(config['conv_filters']*2*2, num_classes)
        
#     def _make_layer(self, blockName, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes:
            
#             downsample = nn.Sequential(
#                 nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride),
#                 nn.BatchNorm1d(planes),
#             )
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)
    
    
#     def forward(self, x):
#         x = x.view(-1, self.nb_channels, self.window_size)
#         x = self.conv1(x)
#         x = self.maxpool(x)
#         x = self.layer0(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         # x = self.layer3(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

#         return x

# class ResNet4(nn.Module):
#     def __init__(self, config, block, layers, num_classes = 2):
#         super(ResNet4, self).__init__()
#         self.nb_channels = config['nb_channels']
#         self.window_size = config['window_size']
#         num_classes = config['nb_classes']       
#         self.inplanes = config['conv_filters']
#         self.conv1 = nn.Sequential(
#                         nn.Conv1d(config['nb_channels'], config['conv_filters'], kernel_size = 7, stride = 2, padding = 3),
#                         nn.BatchNorm1d(config['conv_filters']),
#                         nn.ReLU())
#         self.maxpool = nn.MaxPool1d(kernel_size = 3, stride = 2, padding = 1)
#         self.layer0 = self._make_layer(config['ResNetBlock'], block, config['conv_filters'], layers[0], stride = 1)
#         self.layer1 = self._make_layer(config['ResNetBlock'], block, config['conv_filters']*2, layers[1], stride = 2)
#         self.layer2 = self._make_layer(config['ResNetBlock'], block, config['conv_filters']*2*2, layers[2], stride = 2)
#         self.layer3 = self._make_layer(config['ResNetBlock'], block, config['conv_filters']*2*2*2, layers[2], stride = 2)
#         self.avgpool = nn.AvgPool1d(10, stride=1)
#         self.fc = nn.Linear(config['conv_filters']*2*2*2, num_classes)
        
#     def _make_layer(self, blockName, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes:
            
#             downsample = nn.Sequential(
#                 nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride),
#                 nn.BatchNorm1d(planes),
#             )
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)
    
    
#     def forward(self, x):
#         x = x.view(-1, self.nb_channels, self.window_size)
#         x = self.conv1(x)
#         x = self.maxpool(x)
#         x = self.layer0(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

#         return x


class Classifier(nn.Module):
    def __init__(self, input_size=1024, output_size=2):
        super(Classifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        y_pred = self.linear1(x)
        return y_pred

class EvaClassifier(nn.Module):
    def __init__(self, input_size=1024, nn_size=32, output_size=2, config=None):
        super(EvaClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, config['fc_filters'])
        self.linear2 = torch.nn.Linear(config['fc_filters'], output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

class Downsample(nn.Module):
    r"""Downsampling layer that applies anti-aliasing filters.
    For example, order=0 corresponds to a box filter (or average downsampling
    -- this is the same as AvgPool in Pytorch), order=1 to a triangle filter
    (or linear downsampling), order=2 to cubic downsampling, and so on.
    See https://richzhang.github.io/antialiased-cnns/ for more details.
    """

    def __init__(self, channels=None, factor=2, order=1):
        super(Downsample, self).__init__()
        assert factor > 1, "Downsampling factor must be > 1"
        self.stride = factor
        self.channels = channels
        self.order = order

        # Figure out padding and check params make sense
        # The padding is given by order*(factor-1)/2
        # so order*(factor-1) must be divisible by 2
        total_padding = order * (factor - 1)
        assert total_padding % 2 == 0, (
            "Misspecified downsampling parameters."
            "Downsampling factor and order must be such "
            "that order*(factor-1) is divisible by 2"
        )
        self.padding = int(order * (factor - 1) / 2)

        box_kernel = np.ones(factor)
        kernel = np.ones(factor)
        for _ in range(order):
            kernel = np.convolve(kernel, box_kernel)
        kernel /= np.sum(kernel)
        kernel = torch.Tensor(kernel)
        self.register_buffer(
            "kernel", kernel[None, None, :].repeat((channels, 1, 1))
        )

    def forward(self, x):
        return F.conv1d(
            x,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.shape[1],
        )

class ResBlock(nn.Module):
    r""" Basic bulding block in Resnets:

       bn-relu-conv-bn-relu-conv
      /                         \
    x --------------------------(+)->

    """

    def __init__(
        self, in_channels, out_channels, kernel_size=5, stride=1, padding=2
    ):

        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        x = self.relu(self.bn2(x))
        x = self.conv2(x)

        x = x + identity

        return x


class Resnet(nn.Module):
    r"""The general form of the architecture can be described as follows:

    x->[Conv-[ResBlock]^m-BN-ReLU-Down]^n->y

    In other words:

            bn-relu-conv-bn-relu-conv                        bn-
           /                         \                      /
    x->conv --------------------------(+)-bn-relu-down-> conv ----

    """

    def __init__(
        self,
        output_size=1,
        n_channels=3,
        is_eva=False,
        resnet_version=1,
        epoch_len=10,
        is_mtl=False,
        config=None
    ):
        super(Resnet, self).__init__()

        # Architecture definition. Each tuple defines
        # a basic Resnet layer Conv-[ResBlock]^m]-BN-ReLU-Down
        # isEva: change the classifier to two FC with ReLu
        # For example, (64, 5, 1, 5, 3, 1) means:
        # - 64 convolution filters
        # - kernel size of 5
        # - 1 residual block (ResBlock)
        # - ResBlock's kernel size of 5
        # - downsampling factor of 3
        # - downsampling filter order of 1
        # In the below, note that 3*3*5*5*4 = 900 (input size)
        if resnet_version == 1:
            if epoch_len == 5:
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 2, 2),
                    (256, 5, 2, 5, 3, 1),
                    (256, 5, 2, 5, 3, 1),
                    (512, 5, 0, 5, 3, 1),
                ]
            elif epoch_len == 10:
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 2, 2),
                    (256, 5, 2, 5, 5, 1),
                    (512, 5, 2, 5, 5, 1),
                    (1024, 5, 0, 5, 3, 1),
                ]
            else:
                cgf = [
                    (64, 5, 2, 5, 3, 1),
                    (128, 5, 2, 5, 3, 1),
                    (256, 5, 2, 5, 5, 1),
                    (512, 5, 2, 5, 5, 1),
                    (1024, 5, 0, 5, 4, 0),
                ]
        else:
            cgf = [
                (64, 5, 2, 5, 3, 1),
                (64, 5, 2, 5, 3, 1),
                (128, 5, 2, 5, 5, 1),
                (128, 5, 2, 5, 5, 1),
                (256, 5, 2, 5, 4, 0),
            ]  # smaller resnet
        in_channels = n_channels
        feature_extractor = nn.Sequential()
        for i, layer_params in enumerate(cgf):
            (
                out_channels,
                conv_kernel_size,
                n_resblocks,
                resblock_kernel_size,
                downfactor,
                downorder,
            ) = layer_params
            feature_extractor.add_module(
                f"layer{i+1}",
                Resnet.make_layer(
                    in_channels,
                    out_channels,
                    conv_kernel_size,
                    n_resblocks,
                    resblock_kernel_size,
                    downfactor,
                    downorder,
                ),
            )
            in_channels = out_channels

        self.feature_extractor = feature_extractor
        self.is_mtl = is_mtl

        # Classifier input size = last out_channels in previous layer
        if is_eva:
            self.classifier = EvaClassifier(
                input_size=out_channels, output_size=output_size, config=config
            )
        elif is_mtl:
            self.aot_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
            self.scale_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
            self.permute_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
            self.time_w_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
        else:
            self.classifier = Classifier(
                input_size=out_channels, output_size=output_size
            )

        weight_init(self)

    @staticmethod
    def make_layer(
        in_channels,
        out_channels,
        conv_kernel_size,
        n_resblocks,
        resblock_kernel_size,
        downfactor,
        downorder=1,
    ):
        r""" Basic layer in Resnets:

        x->[Conv-[ResBlock]^m-BN-ReLU-Down]->

        In other words:

                bn-relu-conv-bn-relu-conv
               /                         \
        x->conv --------------------------(+)-bn-relu-down->

        """

        # Check kernel sizes make sense (only odd numbers are supported)
        assert (
            conv_kernel_size % 2
        ), "Only odd number for conv_kernel_size supported"
        assert (
            resblock_kernel_size % 2
        ), "Only odd number for resblock_kernel_size supported"

        # Figure out correct paddings
        conv_padding = int((conv_kernel_size - 1) / 2)
        resblock_padding = int((resblock_kernel_size - 1) / 2)

        modules = [
            nn.Conv1d(
                in_channels,
                out_channels,
                conv_kernel_size,
                1,
                conv_padding,
                bias=False,
                padding_mode="circular",
            )
        ]

        for i in range(n_resblocks):
            modules.append(
                ResBlock(
                    out_channels,
                    out_channels,
                    resblock_kernel_size,
                    1,
                    resblock_padding,
                )
            )

        modules.append(nn.BatchNorm1d(out_channels))
        modules.append(nn.ReLU(True))
        modules.append(Downsample(out_channels, downfactor, downorder))

        return nn.Sequential(*modules)

    def forward(self, x):
        feats = self.feature_extractor(x)

        if self.is_mtl:
            aot_y = self.aot_h(feats.view(x.shape[0], -1))
            scale_y = self.scale_h(feats.view(x.shape[0], -1))
            permute_y = self.permute_h(feats.view(x.shape[0], -1))
            time_w_h = self.time_w_h(feats.view(x.shape[0], -1))
            return aot_y, scale_y, permute_y, time_w_h
        else:
            y = self.classifier(feats.view(x.shape[0], -1))
            return y
        return y


def weight_init(self, mode="fan_out", nonlinearity="relu"):

    for m in self.modules():

        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(
                m.weight, mode=mode, nonlinearity=nonlinearity
            )

        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)