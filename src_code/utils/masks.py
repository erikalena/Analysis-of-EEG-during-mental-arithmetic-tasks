import torch.nn as nn
import torch
import os
import pickle
import numpy as np


class Mask(nn.Module):
    '''
    Class that inherits from torch.nn.Module, implementing the Fourier modulatory mask as a pre-processing layer
    Attribute M (torch.tensor): tensor that stores the entries of the mask
    '''

    def __init__(self, mask_size: tuple = (1, 32, 32)):
        super().__init__()
        assert len(mask_size) == 3
        kernel = torch.ones((1, *mask_size))
        self.M = nn.Parameter(kernel)
        nn.init.ones_(self.M)

    def forward(self, x):
        #x = torch.fft.fft2(x) # fourier transform
        x = self.M * x       # multiply by mask
        #x = torch.fft.ifft2(x).real # inverse fourier transform
        return x
   
   
class MaskedClf(nn.Module):
    '''
    Class that inherits from torch.nn.Module, implementing a end-to-end 'masked' classifier

    Attribute mask (Mask): pre-processing layer doing mask modulation
    Attribute clf (torch.nn.Module): pre-trained classifier
    '''

    def __init__(self, mask, clf):
        super().__init__()
        self.mask=mask
        self.clf=clf

    def forward(self, x):
        x=self.mask(x) # apply mask
        x=self.clf(x)   # apply classifier
        return x



class MaskDataset(torch.utils.data.Dataset):

    '''
    Class that inherits from torch.utils.data.Dataset to store masks

    Input:
    - path (str): path to the masks
    - nclasses (int): number of classes

    Return: a MaskDataset object with the following attributes:
    - attribute masks (torch.tensor): tensor that contains masks
    - attribute labels (torch.tensor): label of each mask
    - attribute id (torch.tensor): id of each mask made of the subject and the task (which is also the label) e.g. S001R03 + 0
    - attribute channel (torch.tensor): channel of each mask

    '''

    def __init__(self, ch=None, path = None, nclasses=None, dict_ch=None):

        self.masks=[]
        self.labels=[]
        self.id=[]
        self.channel=[]

        if path is not None:

            for c in range(nclasses):
                class_list=sorted(os.listdir(path+str(c)))
                
                for mask in class_list:
                    channel = str(mask).split('_')[3][:-4]
                    if ch is not None and channel != ch: # if channel is specified, skip masks that do not belong to that channel
                        continue
                    self.masks.append(np.load(path+str(c)+"/"+mask))
                    self.labels.append(c)
                    id = f'{str(mask)[7:].split("_")[0]}_{str(mask).split("_")[1]}_{str(mask).split("_")[2]}' # subject + label + epoch
                    print(id)
                    self.id.append(id) 
                    self.channel.append(channel)


    
        
    def add_instance(self, elem):
        self.masks.append(elem[0])
        self.labels.append(elem[1])
        self.id.append(elem[2])
        self.channel.append(elem[3])
    
    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        return self.masks[idx], self.labels[idx], self.id[idx], self.channel[idx]