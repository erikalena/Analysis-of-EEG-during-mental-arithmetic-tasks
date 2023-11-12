import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import resnet18, ResNet18_Weights

import matplotlib.pyplot as plt
import time
import os



def load_resnet18(nclasses = 2, pretrained = True, device = 'cpu', input_channels = 1):
    """
    Load a pretrained resnet18 model
    """

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights = weights)

    # change the last layer with the number of classes we have
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, nclasses)
    # add softmax layer
    model = nn.Sequential(model, nn.Softmax(dim=1))

    model = model.to(device)
    model.device = device
    model.input_channels = input_channels   
    
    # change the first layer to accept correct number of channels
    model[0].conv1 = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    return model


class shallow2d(nn.Module):

    def __init__(self, nclasses, input_size, input_channels, device='cpu'):
        super(shallow2d, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 40, kernel_size=5, padding='same') # 40 x Neeg x N
        self.bn1 = nn.BatchNorm2d(40)
        self.conv2 = nn.Conv2d(40, 40, kernel_size=5, padding='valid') # 40 x 1 x N
        self.bn2 = nn.BatchNorm2d(40)

        self.avgpool = nn.AvgPool2d(kernel_size=5)  # 40 x 1 x N/15


        self.input_channels = input_channels

        self.input_size = self.get_in_channels(input_size)
        # fully connected layer
        self.fc1 = nn.Linear(self.input_size, 80)
        self.fc2 = nn.Linear(80, nclasses)
        self.softmax = nn.Softmax(dim=1)

        self.to(device)
        self.device = device


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(torch.nn.ReLU()(x))
        x = self.conv2(x)
        x = self.bn2(torch.nn.ReLU()(x))
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
    def get_in_channels(self, input_size):
        # create dummy input
        x = torch.randn(1, self.input_channels, input_size[0], input_size[1])
        x = self.avgpool(self.bn2(self.conv2(self.bn1(self.conv1(x)))))
        x = torch.flatten(x, start_dim=1)
        return x.shape[1]
 

class ShallowConvNet(nn.Module):
    
    def __init__(self, nclasses, nelectrodes, input_size, input_channels, device='cpu'):
        super(ShallowConvNet, self).__init__()

        
        self.filters = 40
        # two conv layers with 40 kernels per layer
        # first conv layer performs convolution along time axis
        self.conv1 = nn.Conv2d(1, self.filters, kernel_size=(1, 30), padding='same') # 40 x Neeg x N
        # second conv layer performs convolution along electrode axis
        self.conv2 = nn.Conv2d(self.filters, self.filters, kernel_size=(nelectrodes,1), padding='valid') # 40 x 1 x N
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 15))  # 40 x 1 x N/15

        self.input_size = self.get_in_channels(input_size, nelectrodes)

        # fully connected layer
        self.fc1 = nn.Linear(self.input_size, 80)
        self.fc2 = nn.Linear(80, nclasses)
        self.softmax = nn.Softmax(dim=1)

        self.to(device)
        self.device = device
        self.input_channels = input_channels
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
    def get_in_channels(self, input_size, nelectrodes):
        # create dummy input
        x = torch.randn(1, 1, nelectrodes, input_size)
        x = self.avgpool(self.conv2(self.conv1(x)))
        x = torch.flatten(x)
        return x.shape[0]
    

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


def get_weights(dataset, nclasses):
    """
    Function to get weights for each class
    proportional to the inverse of the number of samples
    """

    counts = [sum([1 for i, _ in enumerate(dataset) if dataset.get_label(i) == label]) for label in range(nclasses)]

    # compute class weights
    class_weights = [sum(counts)/(nclasses **2 * counts[i]) for i in range(nclasses)]

    return torch.tensor(class_weights).float()
