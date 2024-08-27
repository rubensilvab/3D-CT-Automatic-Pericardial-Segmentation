# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:52:01 2024

@author: RubenSilva
"""

import pandas as pd
import numpy as np
import os
import pydicom
import nrrd
from torch.utils.data import Dataset
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
class ConvertTo3DnnUnet(Dataset):
    
    def __init__(self, csv_file, path_move):
        
        #CSV File containing a set of patients and the respective path of the dicom and mask
        
        self.csv_file=csv_file
        self.patients = np.unique(csv_file['Patient'])
        self.path_move=path_move
        
    def __len__(self):
        return len(self.patients) #size of the dataset

    def __getitem__(self, idx):
        
        #idx=56
        
        patient=self.csv_file.loc[idx,'Patient']
        fold=self.csv_file.loc[idx,'Fold']
        files_img =self.csv_file.loc[idx,'Path_image']
        files_mask =self.csv_file.loc[idx,'Path_Mask']
        img_size=self.csv_file.loc[idx,'Img_Size']
        imgs_dcm,last_imgpos= self.read_img(files_img,img_size)
        seg_nrrd,head,ordem=self.read_nrrd(files_mask,last_imgpos)
        
        self.plotdcm(imgs_dcm,seg_nrrd)
        
        path_nrrd_save=os.path.join(self.path_move,'imagesTr',str(fold))
        path_nrrd_save_lb=os.path.join(self.path_move,'labelsTr',str(fold))
        
        isExist = os.path.exists(path_nrrd_save)
        if not isExist:                         
            # Create a new directory because it does not exist 
            os.makedirs(path_nrrd_save)
            os.makedirs(path_nrrd_save_lb)
            
        nrrd.write(os.path.join(path_nrrd_save,"OSC_{:03d}_0000".format(idx+1)+'.nrrd'), self.reshape_nrrd(imgs_dcm),header=head)
        nrrd.write(os.path.join(path_nrrd_save_lb,"OSC_{:03d}".format(idx+1)+'.nrrd'), self.reshape_nrrd(seg_nrrd),header=head)
        
        
        return patient,idx,ordem
    

    def plotdcm(self,train_features,train_labels):
        
        for i in range(train_features.shape[0]):
            img = train_features.squeeze()[i]
            label = train_labels.squeeze()[i]
            plt.imshow(img, cmap="gray")
            
            alpha=0.5
            plt.imshow(label,cmap="gray",alpha=alpha)
        
            plt.show()
        
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
           #img = self.normalize(img)
           imgs[i] = img
           
       order = [i for i in range(n_slices)]
       new_order = [i for z, i in sorted(zip(zcoords, order))][::-1] 

       imgs = imgs[new_order].astype(np.int16)
             
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
        print(img_pos,last_imgpos)
        
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
  
#file_path = 'C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/Abdominal_5.csv'
#file_path = 'C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/EPICHEART/EPICHEART_5.csv'
#file_path= 'C:/Users/RubenSilva/Desktop/CardiacCT_peri/CardiacFat/CFAT_5.csv' 
file_path='E:/RubenSilva/PericardiumSegmentation/Dataset/OSIC/OSIC_5.csv'

# Read the CSV file into a DataFrame
csv_file = pd.read_csv(file_path) 
csv_file.loc[:,'Patient']=csv_file['Patient'].astype(str)
csv_file.reset_index(drop=True, inplace=True)

path_save='E:/RubenSilva/PericardiumSegmentation/Dataset/nnUNet_raw/Dataset004_OSC'

dset = ConvertTo3DnnUnet(csv_file,path_save) 
prob_pat=[]
prob_idx=[]

for data in dset:
    
    """Save"""
    patient,idx,ordem=data
    
    print('Patient ',patient,'done.',idx)
    if ordem=='i':
       prob_pat.append(patient)
       prob_idx.append(idx)