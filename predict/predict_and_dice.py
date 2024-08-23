# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:52:52 2024

@author: RubenSilva
"""

# set the matplotlib backend so figures can be saved in the background
import matplotlib
#matplotlib.use("Agg")
# import the necessary packages

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
"""Replace 'file_path.csv' with the path to your CSV file"""

#Choose the folds of the test set
folds_test=[4]

file_path = 'C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/Abdominal_5.csv'

# Read the CSV file into a DataFrame
csv_file = pd.read_csv(file_path) 
csv_file.loc[:,'Patient']=csv_file['Patient'].astype(str)
TestSetcsv=csv_file.loc[csv_file['Fold'].isin(folds_test)] 
TestSetcsv.reset_index(drop=True, inplace=True)
 
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

"""Load the model, choose the path"""
model_path='C:/Users/RubenSilva/Desktop/pytorch_test/models/Mon_Jul_15_17_18_23_2024/Mon_Jul_15_17_18_23_2024.pth'

model_dic=torch.load(model_path)
state_dic=model_dic['model_state_dict']


"""Choose where do you want to save the results"""
dataset='abd_test_csv'
path='E:/RubenSilva/PericardiumSegmentation/Results'
path_results=os.path.join(path,dataset,model_path.split('/')[-2],Posprocess)

model.load_state_dict(state_dic)

with torch.no_grad():
    model.eval()
    
    for i, data in enumerate(Test_dl,0):
        
        """Load data"""
        img, mask, patient=data 
        img, mask = img.to(my_device), mask.to(my_device)
        
        header,img_size=load_header(patient,csv_file)
        
        """Make predictions"""
        pred=predict(model,img,img_size,my_device)
        actual_batch=pred.shape[0]
        
        """Post processment?"""
        
        if pp==1:
            pred= pos_process(pred,my_device)
        
        """Calculate Dice Coefficient"""
        dice=dice_coefficient(pred,mask ,actual_batch,my_device)
        ExcelRowDice(path_results,dice,patient)
        
        """Save 2D images for visialization"""
        save_img_results(path_results,img,pred,mask,dice,patient)
        
        """Save predictions in nrrd"""
        write_nrrd(patient, path_results, header, pred,dataset)
        
        
        print('It remains ',len(Test_dl)-i-1,' patients. Actual patient:',patient[0])
        
        
        # Free up memory
        del img
        del mask
        del pred
       
        # Empty the cache and collect garbage
        torch.cuda.empty_cache()
        gc.collect()
    