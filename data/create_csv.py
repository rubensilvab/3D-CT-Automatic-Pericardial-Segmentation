# -*- coding: utf-8 -*-
"""
Created on Thu May 16 22:54:43 2024

@author: RubenSilva
"""

import os
import glob
import numpy as np
import shutil
import csv
import cv2

"Path of the Dataset"
path='/Data'
os.chdir(path)

#Path of the Dicom files
list_patients=os.listdir(os.path.join(path,"DCM"))
#Path masks
path_nrrd=os.path.join(path,"Peri_segm")

"Number of Folds for cross validation"
number_of_folds=5
patients_p_fold=int(len(list_patients)/number_of_folds)

init_patient=0

"Chose name of the CSV"
name_csv='Data_'
        
"""Write CSV"""

import pydicom

p_prob=[]
total_p_in_fold=patients_p_fold*number_of_folds
npp=0

with open(name_csv+str(number_of_folds)+'.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Patient", "Fold", "Path_image","Path_Mask",'Img_Size',"N_Slices"])

    for fold in range(number_of_folds):
        
        for patient in list_patients[init_patient:init_patient+patients_p_fold]:
            
            n_slices=len(os.listdir(os.path.join(path,"DCM",patient)))
            
            files_dcm=sorted(glob.glob(os.path.join(path,"DCM",patient)+'/*'))
            data = pydicom.read_file(files_dcm[0])
            dcm=data.pixel_array
            img_size=str(dcm.shape)
        
            writer.writerow([str(patient),str(fold), os.path.join(path,"DCM",patient), os.path.join(path_nrrd,patient+".nrrd"),img_size,n_slices])
            
            npp=npp+1
            print(npp,patient)
            
        init_patient=init_patient+patients_p_fold
        
    for p in list_patients[init_patient:]:
            
            #print(path_mask_img)
            n_slices=len(os.listdir(os.path.join(path,"DCM",p)))
            
            files_dcm=sorted(glob.glob(os.path.join(path,"DCM",patient)+'/*'))
            data = pydicom.read_file(files_dcm[0])
            dcm=data.pixel_array
            img_size=str(dcm.shape)
            
            writer.writerow([str(p),str(fold), os.path.join(path,"DCM",p), os.path.join(path_nrrd,p+".nrrd"),img_size,n_slices])
            npp=npp+1
            
            print(npp,'last ones',p)
        