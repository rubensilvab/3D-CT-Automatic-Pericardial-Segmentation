# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:40:19 2024

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

from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
import cv2
from PIL import Image
import pydicom
import glob

#Import Custom DataSet
import os
from torch.utils.data import Dataset
from torchvision.io import read_image

import nrrd as nrrd
import random


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

from torchvision import transforms

class PatientCustomDatasetCSVdcm(Dataset):
    def __init__(self, csv_file,transform=None):
        
        #CSV File containing a set of patients and the respective path of the dicom and mask
        
        self.csv_file=csv_file
        self.transform = transform
        self.patients = np.unique(csv_file['Patient'])
        
    def __len__(self):
        return len(self.patients) #size of the dataset

    def __getitem__(self, idx):
        
        files_img =self.csv_file.loc[idx,'Path_image']
        files_mask =self.csv_file.loc[idx,'Path_Mask']
        img_size=self.csv_file.loc[idx,'Img_Size']
        imgs_dcm= self.read_img(files_img,img_size)
        imgs_mask,_=self.read_nrrd(files_mask)
        
        transform=self.transform 
        
        if self.transform:
            seed = random.randint(0, 2**32)
            
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            imgs_dcm = transform(imgs_dcm)
            #Applied only to the img
            GaussNoise=transforms.RandomApply([AddGaussianNoise(mean=0.0, std=0.05)], p=0.2)
            imgs_dcm=GaussNoise(imgs_dcm)
            
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            imgs_mask = transform(imgs_mask)
            imgs_mask=imgs_mask.to(torch.int8)
        
        imgs_dcm,imgs_mask = torch.unsqueeze(imgs_dcm, 0),torch.unsqueeze(imgs_mask, 0)
        imgs_dcm,imgs_mask = imgs_dcm.float(), imgs_mask.to(torch.int8)
       
        return imgs_dcm, imgs_mask    
    
    def read_img(self, file_patient,img_size):
      
       files=os.listdir(file_patient)
       
       n_slices = len(files)
       img_size=img_size.split(',')
       imgs = np.zeros((n_slices,int(img_size[0][1:]), int(img_size[1][:-1])))
       zcoords = []
       
       for i, file in enumerate(files):
           
           img = pydicom.read_file(os.path.join(file_patient,file))
           zcoords.append(img.get('ImagePositionPatient')[2])
           img = img.pixel_array * img.RescaleSlope + img.RescaleIntercept
           img = self.normalize(img)
           imgs[i] = img
           
       order = [i for i in range(n_slices)]
       new_order = [i for z, i in sorted(zip(zcoords, order))][::-1] 
              
       imgs = imgs[new_order]
       imgs = torch.from_numpy(imgs)
       
       return imgs
   
    def normalize(self, img,img_min=-1000,img_max=1000):
        
        img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level
        img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level
        
        normalize=(img-img_min)/(img_max-img_min)
        
        return normalize
    
    def read_nrrd(self, file_mask):
        
        readdata, header = nrrd.read(file_mask)
        mask=np.array(readdata)
        mask=np.transpose(mask,(2,1,0))
        mask=mask[::-1] 
        mask = torch.from_numpy(mask.copy())
        
        return mask

class PatientCustomTestsetCSVdcm(PatientCustomDatasetCSVdcm):
    def __init__(self, csv_file,transform=None):
        
        #CSV File containing a set of patients and the respective path of the dicom and mask
        self.csv_file=csv_file
        self.transform = transform
        self.patients = np.unique(csv_file['Patient'])

    def __len__(self):
        return len(self.patients) #size of the dataset

    def __getitem__(self, idx):
        
        patient= self.csv_file.loc[idx,'Patient']
        files_img =self.csv_file.loc[idx,'Path_image']
        files_mask =self.csv_file.loc[idx,'Path_Mask']
        img_size=self.csv_file.loc[idx,'Img_Size']
        imgs_dcm= self.read_img(files_img,img_size)
        imgs_mask=self.read_nrrd(files_mask)
         
        if self.transform:
            imgs_dcm = self.transform(imgs_dcm)
            imgs_mask=imgs_mask.to(torch.int8)
        
        imgs_dcm,imgs_mask = torch.unsqueeze(imgs_dcm, 0),torch.unsqueeze(imgs_mask, 0)
        
        imgs_dcm = imgs_dcm.float()
        
        
        return imgs_dcm, imgs_mask, patient

class PatientCustomTestset(PatientCustomDatasetCSVdcm):
    def __init__(self, image_dir,transform=None):
        
        #CSV File containing a set of patients and the respective path of the dicom and mask
        self.image_dir = image_dir
        self.transform = transform
        self.patients = os.listdir(image_dir)

    def __len__(self):
        return len(self.patients) #size of the dataset

    def __getitem__(self, idx):
        
        patient= self.patients[idx]
        files_img =os.path.join(self.image_dir,patient)
        
        #find img_shape
        slices=os.listdir(files_img)
        img = pydicom.read_file(os.path.join(files_img,slices[0]))
        img = img.pixel_array
        img_size=str((img.shape))
        
        imgs_dcm= self.read_img(files_img,img_size)
           
        if self.transform:
            imgs_dcm = self.transform(imgs_dcm)
        
        imgs_dcm = torch.unsqueeze(imgs_dcm, 0)
        
        imgs_dcm = imgs_dcm.float()
        
        
        return imgs_dcm, patient   
    
                 