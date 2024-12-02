# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 6 Jan 2023
author: @vgopakum
Utilities required for training neural-pde surrogate models.
"""
import numpy as np 
import torch 
import torch.nn as nn 
import torch.functional as F 


# %%
##################################
# Normalisation Functions
##################################
def Normalisation(norm_strategy):
    if norm_strategy == 'Min-Max':
        normalizer = MinMax_Normalizer
    elif norm_strategy == 'Range':
        normalizer = RangeNormalizer
    elif norm_strategy == 'Min-Max. Variable':
        normalizer = MinMax_Normalizer_variable
    elif norm_strategy == 'Identity':
        normalizer = Identity
    return normalizer    

# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=-1.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low) / (mymax - mymin)
        self.b = -self.a * mymax + high

    def encode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = self.a * x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = (x - self.b) / self.a
        x = x.view(s)
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()


# normalization, rangewise but each variable at a time.
class MinMax_Normalizer_variable(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(MinMax_Normalizer_variable, self).__init__()
        
        min_u = torch.amin(x, dim=0)
        max_u = torch.amax(x, dim=0)
            
        self.a = (high - low)/(max_u - min_u)
        self.b = -self.a*max_u + high

    def encode(self, x):
        x = self.a*x + self.b
        return x

    def decode(self, x):
        x = (x - self.b)/self.a
        return x
    
    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()


    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()

#normalization, rangewise but across the full domain 
class MinMax_Normalizer(object):
    def __init__(self, x, low=-1.0, high=1.0):
        super(MinMax_Normalizer, self).__init__()
        mymin = torch.min(x)
        mymax = torch.max(x)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()


#normalization, Identity - does nothing
class Identity(object):
    def __init__(self, x, low=-1.0, high=1.0):
        super(Identity, self).__init__()
        self.a = torch.tensor(0)
        self.b = torch.tensor(0)

    def encode(self, x):
        return x 

    def decode(self, x):
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()
