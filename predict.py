# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:52:52 2024

@author: RubenSilva
"""

# set the matplotlib backend so figures can be saved in the background
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
from New_CustomDataset import PatientCustomTestsetCSVdcm
import openpyxl
from functions import *
from PosProcess import *

if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))  


from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
])

import pandas as pd    


"""Define Test sets""" 
# Replace 'file_path.csv' with the path to your CSV file
# folds_test=[0,1,2,3,4]
# file_path = 'C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/EPICHEART/EPICHEART_5.csv'

folds_test=[4]

file_path = 'C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/Abdominal_5.csv'

#file_path= 'C:/Users/RubenSilva/Desktop/CardiacCT_peri/CardiacFat/CFAT_5.csv' 

# file_path='E:/RubenSilva/PericardiumSegmentation/Dataset/OSIC/OSIC_5.csv'

if 'EPICHEART' in file_path:
    dataset= 'EPICHEART'
elif 'Abdominal' in file_path:
    dataset='Abdominal'
elif 'CFAT' in file_path: 
    dataset='CardiacFat'
else:
    dataset='OSIC'

# Read the CSV file into a DataFrame
csv_file = pd.read_csv(file_path) 
csv_file.loc[:,'Patient']=csv_file['Patient'].astype(str)
TestSetcsv=csv_file.loc[csv_file['Fold'].isin(folds_test)] 
TestSetcsv.reset_index(drop=True, inplace=True)
 


"""Só depois de um certo paciente, para caso que por alguma razão falhe até px"""
# patients=sorted(TestSetcsv['Patient'])
# pat_begin=patients[241:]
# TestSetcsv=csv_file.loc[csv_file['Patient'].isin(pat_begin)]

"""Apply DataLoader"""
TestDataset = PatientCustomTestsetCSVdcm(TestSetcsv,transform=transform)
Test_dl= DataLoader(TestDataset, batch_size=1, shuffle=False)

"""Post processesment?"""
pp=1 

if pp==1:
    Posprocess='WithPP'
else:
    Posprocess='WithoutPP'

"""Import model"""
from can3d_multi_noskip import CAN3D 
import gc

print("[INFO] initializing the CAN3D model...")
model=CAN3D().to(my_device)

model_path='C:/Users/RubenSilva/Desktop/pytorch_test/models/Mon_Jul_15_17_18_23_2024/Mon_Jul_15_17_18_23_2024.pth'

model_dic=torch.load(model_path)
state_dic=model_dic['model_state_dict']

path_results=os.path.join('E:/RubenSilva/PericardiumSegmentation/Results',dataset,model_path.split('/')[-2],Posprocess)

model.load_state_dict(state_dic)

with torch.no_grad():
    model.eval()
    
    for i, data in enumerate(Test_dl,0):
        
        """Load data"""
        img, mask, patient=data 
        img, mask = img.to(my_device), mask.to(my_device)
        
        """Make predictions"""
        pred=predict(model,img,mask,my_device)
        actual_batch=pred.shape[0]
        
        """Remove padding (if necessary)"""
        # indices_to_remove = torch.where(torch.all(img == 0, dim=(0, 1, 3, 4)))[0]
        
        # # Create a mask to keep indices not in indices_to_remove
        # mask_remove = torch.ones(img.shape[2], dtype=bool)
        # mask_remove[indices_to_remove] = False

        # # Use the mask to filter the tensor
        # img= img[:, :, mask_remove, :, :]
        # mask= mask[:, :, mask_remove, :, :]
        # pred= pred[:, :, mask_remove, :, :]
        
        """Post processment?"""
        
        if pp==1:
            pred= pos_process(pred,my_device)
        
        """Calculate Dice Coefficient"""
        dice=dice_coefficient(pred,mask ,actual_batch,my_device)
        ExcelRowDice(path_results,dice,patient)
        #dice=torch.tensor([0])
        save_img_results(path_results,img,pred,mask,dice,patient)
        write_nrrd(patient, path_results, mask, pred,dataset)
        
        
        print('It remains ',len(Test_dl)-i-1,' patients. Actual patient:',patient[0])
        
        
        # Free up memory
        del img
        del mask
        del pred
       
        # Empty the cache and collect garbage
        torch.cuda.empty_cache()
        gc.collect()
    