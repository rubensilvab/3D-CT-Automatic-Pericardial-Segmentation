# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 11:10:43 2024

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
from New_CustomDataset import PatientCustomTestset
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

"""Import model"""
from can3d_multi_noskip import CAN3D 
import gc

print("[INFO] initializing the CAN3D model...")
model=CAN3D().to(my_device)

"""Choose model"""
model_path='C:/Users/RubenSilva/Desktop/pytorch_test/models/Mon_Jul_15_17_18_23_2024/Mon_Jul_15_17_18_23_2024.pth'

model_dic=torch.load(model_path)
state_dic=model_dic['model_state_dict']

model.load_state_dict(state_dic)

"""Choose input folder"""

img_dir='C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/EPICHEART/dcm'

"""Apply DataLoader"""
TestDataset = PatientCustomTestset(img_dir,transform=transform)
Test_dl= DataLoader(TestDataset, batch_size=1, shuffle=False)

"""Post processesment?"""
pp=1 

if pp==1:
    Posprocess='WithPP'
else:
    Posprocess='WithoutPP'

"""Choose output folder"""

path_output='E:/RubenSilva/PericardiumSegmentation/Results'
dataset='epi_test'
path_results=os.path.join(path_output,dataset,model_path.split('/')[-2],Posprocess)


with torch.no_grad():
    model.eval()
    
    for i, data in enumerate(Test_dl,0):
        
        """Load data"""
        img, patient=data 
        img = img.to(my_device)
        
        """Create header for NRRD"""
        header,img_size=create_header(patient,img_dir)
        
        """Make predictions"""
        pred=predict(model,img,img_size,my_device)
        actual_batch=pred.shape[0]
        
        """Post processment?"""
        
        if pp==1:
            pred= pos_process(pred,my_device)
        
        """Save results"""
        
        write_nrrd(patient, path_results, header, pred,dataset)
             
        print('It remains ',len(Test_dl)-i-1,' patients. Actual patient:',patient[0])
        
        
        # Free up memory
        del img
        del pred
       
        # Empty the cache and collect garbage
        torch.cuda.empty_cache()
        gc.collect()
    