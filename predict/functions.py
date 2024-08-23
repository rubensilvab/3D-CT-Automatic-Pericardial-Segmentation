# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:40:07 2024

@author: RubenSilva
"""

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
    
    path_patient=os.path.join(path,str(patient[0]),'Images2D')
    isExist = os.path.exists(path_patient)
    if not isExist:                         
        # Create a new directory because it does not exist 
        os.makedirs(path_patient)  
    os.chdir(path_patient)   
    s=0
    
    print('Saving 2D images predictions...')
    for i in range (pred.shape[2]):
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
        fig.savefig('Predict_'+str(patient[0])+"_"+str(s)+'.jpg')
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
    sheet.append([patient[0], dice.item()])
    #print([patient, slic, label, pred])
    # Save the workbook to a file
    
    book.save(filename)
    
def reshape_nrrd(data):
  nrrd=data[::-1] 
  nrrd=np.transpose(nrrd,(2,1,0))  
 
  return nrrd     

import nrrd

def RealThickness(dcm1,dcm2):
    th1=dcm1.ImagePositionPatient[-1]
    th2=dcm2.ImagePositionPatient[-1]
    realthk=abs(th1-th2)
        
    return realthk

def OrderSlices(files,path_patient):
    
    n_slices=len(files)
    slices=[]
    zcoords=[]
    for i, file in enumerate(files):
        
        img = pydicom.read_file(os.path.join(path_patient,file))
        zcoords.append(img.get('ImagePositionPatient')[2])
        slices.append(file)
        
    order = [i for i in range(n_slices)]
    new_order = [i for z, i in sorted(zip(zcoords, order))][::-1] 
    
    reordered_slices = [slices[i] for i in new_order]
    
    return reordered_slices

import pydicom

def load_header(patient,csv_file):
    
    row_patient=csv_file.loc[csv_file['Patient'] == patient[0]]
    file_mask =row_patient['Path_Mask'].iloc[0]
    img_size=row_patient['Img_Size'].iloc[0]
    img_size=img_size.split(',')
    img_size = [int(img_size[0][1:]), int(img_size[1][:-1])]
    
    _, header = nrrd.read(file_mask)
    
    return header,img_size
    
def create_header(patient,image_dir):
    
    file_patient =os.path.join(image_dir,patient[0])
    
    #find img_shape
    slices=os.listdir(file_patient)
    img = pydicom.read_file(os.path.join(file_patient,slices[0]))
    img = img.pixel_array
    img_size=[img.shape[0],img.shape[1]]
    
    files=os.listdir(file_patient)
    files_dcm=OrderSlices(files,file_patient)
    
    # Extract metadata from the first and scd DICOM file
    first_dicom,scd_dicom = pydicom.read_file(os.path.join(file_patient,files_dcm[0])),pydicom.read_file(os.path.join(file_patient,files_dcm[1]))
    spacing = [float(first_dicom.PixelSpacing[0]), float(first_dicom.PixelSpacing[1]), float(RealThickness(first_dicom,scd_dicom))]
    
    last_dicom=pydicom.read_file(os.path.join(file_patient,files_dcm[-1]))
    
    # Create NRRD header
    o_img_shape=(len(files),img_size[0],img_size[1])
    
    header = {
        'type': 'uint8',
        'dimension': 3,
        'sizes':o_img_shape,
        'space': 'left-posterior-superior',
        'space directions': np.diag(spacing),
        'kinds': ['domain', 'domain', 'domain'],
        'endian': 'little',
        'encoding': 'gzip',
        'space origin': last_dicom.ImagePositionPatient
    }
    
    return header,img_size

def write_nrrd(patient,path,header,pred, dataset):
    
    pred=np.squeeze(pred.cpu().numpy())
        
    path_patient=os.path.join(path,str(patient[0]),'NRRD')
    isExist = os.path.exists(path_patient)
    if not isExist:                         
        # Create a new directory because it does not exist 
        os.makedirs(path_patient)  
    os.chdir(path_patient)  
    
    print('Writing NRRD...')
    nrrd.write(str(patient[0])+'_pred'+'.nrrd', reshape_nrrd(pred),header=header)
    
def predict(model,img,img_size, device):
    #make the predictions and calculate the validation loss
    pred = model(img)
    img_shape=img.shape
    # Create a new torch.Size object with updated dimensions
    new_img_shape = torch.Size(img_shape[:3] + (img_size[0], img_size[1]))
    
    predict = torch.zeros(new_img_shape).to(device)
    
    actual_batch=pred.shape[0]
    for b in range(actual_batch):
    
        predict[b]= F.interpolate(pred[b], size=(img_size[0], img_size[1]), mode='bilinear')
        predict[b]= (predict[b] >= 0.5).to(torch.uint8)
        
    return predict.to(torch.uint8)    
