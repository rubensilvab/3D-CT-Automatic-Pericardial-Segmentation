# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 14:15:36 2024

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
from functionsnnUnet import *
from PosProcess import *

if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))  


import pandas as pd    

class ReConvert3DnnUnet(Dataset):
    
    def __init__(self, csv_file, path_move):
        
        #CSV File containing a set of patients and the respective path of the dicom and mask
        
        self.csv_file=csv_file
        self.patients = np.unique(csv_file['Patient'])
        self.path_move=path_move
        
    def __len__(self):
        return len(self.patients) #size of the dataset

    def __getitem__(self, ind):
        
        #idx=56
        path_pred_nnunet='E:/RubenSilva/PericardiumSegmentation/Dataset/nnUNet_raw/Dataset003_CFT/prediction'
        
        idx = self.csv_file.index[ind]
        
        patient=self.csv_file.loc[idx,'Patient']
        fold=self.csv_file.loc[idx,'Fold']
        files_img =self.csv_file.loc[idx,'Path_image']
        files_mask =self.csv_file.loc[idx,'Path_Mask']
        img_size=self.csv_file.loc[idx,'Img_Size']
        imgs_dcm,last_imgpos= self.read_img(files_img,img_size)
        seg_nrrd,head,ordem=self.read_nrrd(files_mask,last_imgpos)
        
        file_nnunet=os.path.join(path_pred_nnunet,"CFT_{:03d}".format(idx+1)+'.nrrd')
        pred_nrrd,_,_=self.read_nrrd(file_nnunet,last_imgpos)
        
        #self.plotdcm(imgs_dcm,seg_nrrd)
        
        path_nrrd_save=os.path.join(self.path_move,patient)
        
        isExist = os.path.exists(path_nrrd_save)
        if not isExist:                         
            # Create a new directory because it does not exist 
            os.makedirs(path_nrrd_save)
            
        nrrd.write(os.path.join(path_nrrd_save,patient+'_manual.nrrd'), self.reshape_nrrd(seg_nrrd),header=head)
        nrrd.write(os.path.join(path_nrrd_save,patient+'_pred.nrrd'), self.reshape_nrrd(pred_nrrd),header=head)
        
        imgs_dcm= torch.from_numpy(imgs_dcm)
        seg_nrrd=torch.from_numpy(seg_nrrd.copy())
        pred_nrrd =torch.from_numpy(pred_nrrd.copy())
        
        imgs_dcm,seg_nrrd,pred_nrrd = torch.unsqueeze(imgs_dcm, 0),torch.unsqueeze(seg_nrrd, 0),torch.unsqueeze(pred_nrrd, 0)
        
        imgs_dcm = imgs_dcm.float()
        
        return imgs_dcm,seg_nrrd,pred_nrrd,patient
    

    # def plotdcm(self,train_features,train_labels):
        
    #     for i in range(train_features.shape[0]):
    #         img = train_features.squeeze()[i]
    #         label = train_labels.squeeze()[i]
    #         plt.imshow(img, cmap="gray")
            
    #         alpha=0.5
    #         plt.imshow(label,cmap="gray",alpha=alpha)
        
    #         plt.show()
        
    def read_img(self, file_patient,img_size):
       #print(file_patient)
       files=os.listdir(file_patient)
       
       #print(files)
       n_slices = len(files)
       img_size=img_size.split(',')
       imgs = np.zeros((n_slices,int(img_size[0][1:]), int(img_size[1][:-1])))
       zcoords = []
       imgpostpat=[]
       for i, file in enumerate(files):
           
           img = pydicom.read_file(os.path.join(file_patient,file))
           zcoords.append(img.get('ImagePositionPatient')[2])
           imgpostpat.append(img.get('ImagePositionPatient'))
           img = img.pixel_array * img.RescaleSlope + img.RescaleIntercept
           img = self.normalize(img)
           imgs[i] = img
           
       order = [i for i in range(n_slices)]
       new_order = [i for z, i in sorted(zip(zcoords, order))][::-1] 

       imgs = imgs[new_order]
             
       imgpostpat = [imgpostpat[i] for i in new_order]
       last_img=imgpostpat[-1]
       
       return imgs,last_img
   
    #def create_DCMheader(dcm):
        
    def normalize(self, img,img_min=-1000,img_max=1000):
        
        img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level
        img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level
        
        normalize=(img-img_min)/(img_max-img_min)
        
        return normalize
    
    def read_nrrd(self, file_mask,last_imgpos):
        
        readdata, header = nrrd.read(file_mask)
        mask=np.array(readdata)
        mask=np.transpose(mask,(2,1,0))
        img_pos=header['space origin']
        #print(img_pos,last_imgpos)
        
        if round(img_pos[-1],1)-round(last_imgpos[-1],1)<12:
            mask=mask[::-1] 
            ordem='n'
        else:
            print('Ordem inversa')
            ordem='i'
        
        return mask,header,ordem
    
    def reshape_nrrd(self,data):
        
      nrrd=data[::-1] 
      nrrd=np.transpose(nrrd,(2,1,0))  
     
      return nrrd 

import gc

"""Define Test sets""" 
# Replace 'file_path.csv' with the path to your CSV file

# folds_test=[0,1,2,3]
# file_path = 'C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/EPICHEART/EPICHEART_5.csv'

folds_test=[4]

#file_path = 'C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/Abdominal_5.csv'
file_path= 'C:/Users/RubenSilva/Desktop/CardiacCT_peri/CardiacFat/CFAT_5.csv' 
#file_path='E:/RubenSilva/PericardiumSegmentation/Dataset/OSIC/OSIC_5.csv'

# Read the CSV file into a DataFrame
csv_file = pd.read_csv(file_path) 
csv_file.loc[:,'Patient']=csv_file['Patient'].astype(str)
csv_file=csv_file.loc[csv_file['Fold'].isin(folds_test)] 

path_save='E:/RubenSilva/PericardiumSegmentation/Results/CardiacFat/3DnnUnet'

dset = ReConvert3DnnUnet(csv_file,path_save) 

"""Post processesment?"""
pp=0 

if pp==1:
    Posprocess='WithPP'
else:
    Posprocess='WithoutPP'


for i,data in enumerate(dset,0):
    
    """Load data"""
    img, mask,pred,patient=data 
    img, mask,pred = img.to(my_device), mask.to(my_device),pred.to(my_device)
    
    actual_batch=pred.shape[0]
   
    """Post processment?"""
    
    if pp==1:
        pred= pos_process(pred,my_device)
    
    """Calculate Dice Coefficient"""
    dice=dice_coefficient(pred,mask ,actual_batch,my_device)
    ExcelRowDice(path_save,dice,patient)
    save_img_results(path_save,img,pred,mask,dice,patient)
    
    
    print('It remains ',len(dset)-i-1,' patients. Actual patient:',patient)   
        
    # Free up memory
    del img
    del mask
    del pred
   
    # Empty the cache and collect garbage
    torch.cuda.empty_cache()
    gc.collect()
    