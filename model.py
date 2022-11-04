#!/usr/bin/env python

import torch
import torch.nn as nn

class CNN(nn.Module):  # input: 224 x 224 x 3
    def __init__(self, image_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=0)
        # self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=7, stride=2, padding=0)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=7, stride=2, padding=0)
        # self.bn3 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1*16, num_classes)

    def forward(self, x): 
        x = self.conv1(x.float())
        # x = self.bn1(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1) 
        x = self.fc(x)
        return x

    