# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:40:19 2024

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
        imgs_mask=self.read_nrrd(files_mask)
        
        transform=self.transform 
        
        if self.transform:
            seed = random.randint(0, 2**32)
            
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            imgs_dcm = transform(imgs_dcm)
            #Applied ontly to the img
            GaussNoise=transforms.RandomApply([AddGaussianNoise(mean=0.0, std=0.05)], p=0.2)
            imgs_dcm=GaussNoise(imgs_dcm)
            
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            imgs_mask = transform(imgs_mask)
            imgs_mask=imgs_mask.to(torch.int8)
        
   
        #imgs_dcm,imgs_mask =self.limit_slices(imgs_dcm, imgs_mask)
        imgs_dcm,imgs_mask = torch.unsqueeze(imgs_dcm, 0),torch.unsqueeze(imgs_mask, 0)
        imgs_dcm,imgs_mask = imgs_dcm.float(), imgs_mask.to(torch.int8)
        
        #print('New',patient,len(np.unique(imgs_dcm)))
        #print('New, mask',patient,np.unique(imgs_mask))
        #imgs_dcm = imgs_dcm.to(torch.float16)
        
        return imgs_dcm, imgs_mask    
    
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
  
           
def RandomCrop(imgs,mask):
    
    min_slices = 30
    apply_crop = random.randint(0, 1)   # 50% probability
    z_dim = imgs.shape[2]
    
    if apply_crop==1 and (z_dim > min_slices):
        
        #print('crop')
        upper_slices = random.randint(1, int(z_dim * 0.1))
        lower_slices = random.randint(1, int(z_dim * 0.1))
    
        imgs = imgs[:,:,upper_slices:-lower_slices,:,:]
        mask =  mask[:,:,upper_slices:-lower_slices,:,:]
        
        return (imgs,mask)
        
    else:
        
        return (imgs,mask)

    
import pandas as pd    

class SubsetSampler():
    
    def __init__(self, indices,nslices,nslicesind,batch_size, generator=None) -> None:
            self.indices = indices # [0 1 2 3 4]
            self.nslices = nslices # [3 4 3 3 4]
            self.nslicesind = nslicesind # [[], [], [], [1 3 4], [2 5]]
            self.generator = generator
            self.bs = batch_size
    
    def __len__(self):
        return len(self.indices)
    
    def __iter__(self):
        batch=np.zeros((self.bs))
        for i in torch.randperm(len(self.indices)):
            n =self.nslices[i]
            #batch=[self.indices[i]]
            batch[0]=self.indices[i]
            if self.nslicesind[n]:  # Ensure there is at least 1 sample 
                for b in range(self.bs -1):
                    ii = np.random.choice(self.nslicesind[n])
                    #batch.append(self.indices[ii])
                    batch[b+1]=ii
                    
            #print(batch)        
            yield batch        
                
from torchvision import transforms

def ExtractSliceInd(df): 
    # Directly extract the 'N_Slices' column
    N_slices = df['N_Slices'].tolist()
    indices = df.index.tolist()

    # Initialize SliPind with empty lists for each unique value in N_slices
    SliPind = [[] for _ in range(max(N_slices) + 1)]

    # Populate SliPind using a loop
    for idx, slc in zip(indices, N_slices):
        SliPind[slc].append(idx)

    return indices,N_slices,SliPind

#         plt.imshow(label,cmap="gray",alpha=alpha)
    
#         plt.show()
        
        