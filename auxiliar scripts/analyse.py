# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:16:27 2024

@author: RubenSilva
"""

import os
import glob
import numpy as np

import cv2
import matplotlib.pyplot as plt
import pydicom

PATH_X="C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/DICOM/"
PATH_Y="C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/PERI_segm_tif/"

PATH_DCM='E:/CHVNGE_CT/Abdominal/Data_CCT/'
list_patients=sorted(os.listdir(PATH_Y))

n=1
# for patient in list_patients:
#     files_x=sorted(glob.glob(PATH_X+patient+'/*'))
#     files_y=sorted(glob.glob(PATH_Y+patient+'/*'))
    

#     #pick the middle slice to compare
#     slice_m=int(len(files_x)/2)
#     print("slice m:",slice_m,"patient: ", patient, n)
#     fig=plt.figure(figsize=(10,10))
#     img_x=cv2.imread(files_x[slice_m],0)
#     img_y=cv2.imread(files_y[slice_m],0)
#     plt.subplot(1,2,1)
#     plt.imshow(img_x,cmap='gray')
#     plt.title(str(patient)+" s: "+str(slice_m))
#     plt.subplot(1,2,2)
#     plt.imshow(img_y,cmap='gray')
#     plt.title(str(patient)+" s: "+str(slice_m))
    
#     n=n+1
    
"""Detect error"""   
# patients_prob=[] 

# slices=[]
# for patient in list_patients:
#     files_x=sorted(glob.glob(PATH_X+patient+'/*'))
#     files_y=sorted(glob.glob(PATH_Y+patient+'/*'))
    
#     print(len(files_x),len(files_y))
#     slices.append(len(files_x))

#     #pick the middle slice to compare
#     slice_m=int(len(files_x)/2)
    
    # #fig=plt.figure(figsize=(10,10))
    # img_x=cv2.imread(files_x[slice_m],0)
    # img_y1=cv2.imread(files_y[slice_m],0)
    # img_y2=cv2.imread(files_y[slice_m+1],0)
    
    # if (np.sum(img_y1)==0 or  np.sum(img_y2)==0):
        
        # patients_prob.append(patient)
        # plt.subplot(1,2,1)
        # plt.imshow(img_x,cmap='gray')
        # plt.title(str(patient)+" s: "+str(slice_m))
        # plt.subplot(1,2,2)
        # plt.imshow(img_y1,cmap='gray')
        # plt.title(str(patient)+" s: "+str(slice_m))
        # print("slice m:",slice_m,"patient: ", patient, n)
        # n=n+1    
     
    # if (len(files_x) != len(files_y)):
        
    #     patients_prob.append(patient)
    #     plt.subplot(1,2,1)
    #     plt.imshow(img_x,cmap='gray')
    #     plt.title(str(patient)+" s: "+str(slice_m))
    #     plt.subplot(1,2,2)
    #     plt.imshow(img_y1,cmap='gray')
    #     plt.title(str(patient)+" s: "+str(slice_m))
    #     print("slice m:",slice_m,"patient: ", patient, n)
    #     n=n+1    
    #     print(patient)

"""Detect error"""  

# def RealThickness(dcm1,dcm2):
#     th1=float(dcm1.ImagePositionPatient[-1])
#     th2=float(dcm2.ImagePositionPatient[-1])
#     realthk=th1-th2
#     realthk = round(realthk, 3)
    
#     return realthk
 
# patients_thikness=[] 
# patients_prob_th=[]
# patients_prob_inv=[]

# #list_patients=['102001']

# for patient in list_patients:
#     files_x=sorted(glob.glob(PATH_DCM+patient+'/*'))
    
#     for i in range(len(files_x)):
#         try:
#             # manufacturer, pixel spacinf, thikness
#             dicom_data1 = pydicom.read_file(files_x[i])
#             dicom_data2 = pydicom.read_file(files_x[i+1])
            
#             realth=RealThickness(dicom_data1,dicom_data2)
#             patients_thikness.append(realth)
            
#         except:
#              pass
    
#     #print('Patient:',patient, np.unique(patients_thikness))
#     if len(np.unique(patients_thikness))>1:
#         patients_prob_th.append(patient)
#         print('Many Thikcness Patient:',patient)
#     if np.unique(patients_thikness)[0] <0:
#         patients_prob_inv.append(patient)
#         print('problem order Patient:',patient)
        
#     patients_thikness=[]     

"""Analyse patients with more than 64 slices pericardium and respective thickness"""

import pandas as pd
# Replace 'file_path.csv' with the path to your CSV file
file_path = 'CHVNGE/npericardium.xlsx'
dcm_path='CHVNGE/dcmInfo.xlsx'
# Read the CSV file into a DataFrame
df = pd.read_excel(file_path)
dcm= pd.read_excel(dcm_path)

pericardium_p = df.iloc[:,: 2]

dic={}
for i in range(len(pericardium_p)):
    if pericardium_p.iloc[i,1]>64:
        pat=pericardium_p.iloc[i,0]
        ind=dcm.index[dcm['Patient']==pat]
        dic[pat]= dcm['Slice Thickness'].iloc[ind]
        print(pat,dcm['Slice Thickness'].iloc[ind])
        
        
    
        