# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:49:30 2024

@author: RubenSilva
"""


import matplotlib
#matplotlib.use("Agg")
# import the necessary packages
from LeNet import LeNet
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
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
from CustomDatasets import PatientCustomTestsetCSV
import openpyxl


def dice_coefficient(output,target, batch_size,device):
    
    loss = torch.zeros(batch_size).to(device)

    for i in range(batch_size):
        #dice_foreground = torch.sum(target[i] * output[i]) / (torch.sum(target[i]**2) + torch.sum(output[i]**2) + 1e-7)
        smooth = 1e-8
        output_flat = output[i].view(-1)
        target_flat = target[i].view(-1)
        intersection = torch.sum(target_flat * output_flat)
        union = torch.sum(output_flat) + torch.sum(target_flat)
        dice_foreground = (2. * intersection) / (union + smooth)
        
        loss[i] = dice_foreground #- dice_background

    return loss.mean()

def save_img_results(path,img,pred,mask,dice,patient):
    
    path_patient=os.path.join(path,str(patient),'Images2D')
    isExist = os.path.exists(path_patient)
    if not isExist:                         
        # Create a new directory because it does not exist 
        os.makedirs(path_patient)  
    os.chdir(path_patient)   
    s=0
    
    print('Saving 2D images predictions...')
    for i in range (pred.shape[1]):
        s=s+1 
        fig=plt.figure(figsize=(16,6))
        fig.suptitle('Dice:'+str(round(dice.item(),4)))
        plt.subplot(1,3,1)
        plt.imshow(np.squeeze(img.cpu().numpy())[i],cmap='gray',vmin=0,vmax=1)
        plt.title('Original_'+str(s))
        plt.subplot(1,3,2)
        plt.imshow(np.squeeze(mask.cpu().numpy())[i],cmap='gray',vmin=0,vmax=1)
        plt.title('label Test_'+str(s))
        plt.subplot(1,3,3)
        plt.imshow(np.squeeze(pred.cpu().numpy())[i],cmap='gray',vmin=0,vmax=1)
        plt.title('Predict_'+str(s))
        #plt.colorbar()
        fig.savefig('Predict_'+str(patient)+"_"+str(s)+'.jpg')
        plt.close('all')
    
    
      
def ExcelRowDice(path,dice,patient):
    
    isExist = os.path.exists(path)
    if not isExist:                         
        # Create a new directory because it does not exist 
        os.makedirs(path)  
    os.chdir(path)   

    name='Dice'        
    filename = str(name)+'.xlsx'
    
    print('Saving evaluation metrics to excel...')
    # Check if the file already exists
    if not os.path.isfile(filename):
        # Create a new workbook object
        book = openpyxl.Workbook()
        # Select the worksheet to add data to
        sheet = book.active
        # Add a header row to the worksheet
        sheet.append(['Patient', 'Dice'])
    else:
        # Open an existing workbook
        book = openpyxl.load_workbook(filename)
    
        # Select the worksheet to add data to
        sheet = book.active
    
    # Append a new row of data to the worksheet
    sheet.append([patient, dice.item()])
    #print([patient, slic, label, pred])
    # Save the workbook to a file
    
    book.save(filename)
    
def reshape_nrrd(data):
  nrrd=data[::-1] 
  nrrd=np.transpose(nrrd,(2,1,0))  
 
  return nrrd     

import nrrd

