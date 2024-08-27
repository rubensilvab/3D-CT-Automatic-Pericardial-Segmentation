# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:21:02 2024

@author: RubenSilva
"""

import matplotlib
#matplotlib.use("Agg")
# import the necessary packages


import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
import cv2
from PIL import Image

#Import Custom DataSet

import os
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.io import read_image

import openpyxl
import nrrd


path='//10.227.103.133/ChestXRay/RubenSilva/3drui/7252'
mask_path=os.path.join(path,'7252_manual.nrrd')
pred_mask=os.path.join(path,'7252_pred.nrrd')

def read_nrrd(file_mask):
    
    readdata, header = nrrd.read(file_mask)
    mask=np.array(readdata)
    mask=np.transpose(mask,(2,1,0))
    mask[mask > 0] = 1.
    mask[mask >= .5] = 1.
    mask[mask < .5] = 0.
    mask=mask[::-1] 
    mask = mask.astype(np.int16)
    
    mask = torch.from_numpy(mask.copy())
    
    return mask

def dice_coefficient(output,target, batch_size):

    loss = torch.zeros(batch_size)

    
    smooth = 1e-8
    output_flat = output.view(-1)
    target_flat = target.view(-1)
    intersection = torch.sum(target_flat * output_flat)
    union = torch.sum(output_flat) + torch.sum(target_flat)
    dice_foreground = (2. * intersection) / (union + smooth)
    
    loss[0] = dice_foreground #- dice_background

    return loss.mean()

def calcDiceJaccard(mask1, mask2):
    union = ((mask1 + mask2) > 0).sum()
    inter = ((mask1 + mask2) == 2).sum()

    return inter / union, 2 * inter / (mask1.sum() + mask2.sum())


mask=read_nrrd(mask_path)
pred=read_nrrd(pred_mask)

print(dice_coefficient(mask,pred, 1))
print(calcDiceJaccard(mask, pred))
