# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:33:49 2024

@author: RubenSilva
"""

import nrrd
import matplotlib.pyplot as plt

import pydicom 
import numpy as np

import os
import glob

import cv2
import shutil
import openpyxl


# All_folds='E:/RubenSilva/PericardiumSegmentation/Dataset/nnUNet_raw'
# datasets=os.listdir(All_folds)

# for dtaset in datasets:
    
#     #Pasta q quero copiar
    
#     path_fold=os.path.join(All_folds,dtaset)
    
#     #Pasta onde queres guardar os nrrd
#     path_move="E:/RubenSilva/PericardiumSegmentation/Dataset/nnUnet_raw_all/Dataset005_ALL"
    
#     shutil.copytree(path_fold,path_move,dirs_exist_ok=True) 
#     print(path_fold) 

"Rename"
# import re
# Path='E:/RubenSilva/PericardiumSegmentation/Dataset/nnUnet_raw_all/Dataset005_ALL'
# All_fold='E:/RubenSilva/PericardiumSegmentation/Dataset/nnUnet_raw_all/Dataset005_ALL/imagesTr'
# folds=os.listdir(All_fold)
# idx=1
# for fold in folds:
    
#     #Pasta q quero copiar 
#     path_fold_images=os.path.join(Path,'imagesTr',fold)
#     path_fold_labels=os.path.join(Path,'labelsTr',fold)
#     #Pasta onde queres guardar os nrrd
#     path_move="E:/RubenSilva/PericardiumSegmentation/Dataset/nnUnet_raw_all/Dataset005_ALL"
    
#     files=glob.glob(os.path.join(path_fold_images,'*'))
#     for file in files:
#         old_name=file.split('\\')[-1]
#         old_name_label=re.sub(r'_0000(?=\.\w+$)', '', old_name)
        
#         new_name="ALL_{:03d}_0000".format(idx)+'.nrrd'
#         new_name_label="ALL_{:03d}".format(idx)+'.nrrd'
        
#         path_new_image=os.path.join(path_fold_images,new_name)
#         path_new_label=os.path.join(path_fold_labels,new_name_label)
#         path_old_label=os.path.join(path_fold_labels,old_name_label)
        
#         #print(file)
#         print('image:',file,path_new_image)
#         print('label:',path_old_label,path_new_label)
#         os.rename(file, path_new_image)
#         os.rename(path_old_label,path_new_label)
#         idx=idx+1
    
    # shutil.copytree(path_fold,path_move,dirs_exist_ok=True) 
    # print(path_fold)