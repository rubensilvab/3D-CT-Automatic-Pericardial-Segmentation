# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:33:18 2024

@author: RubenSilva
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 21:50:00 2023

@author: RubenSilva
"""



import nrrd
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pydicom 
import numpy as np
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
import os
import glob
#from tensorflow.keras.preprocessing.image import load_img
#from tensorflow.keras.preprocessing.image import save_img
#from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import shutil
import openpyxl




"""
- Pasta principal(teste)
         - Paciente 1
                 - slice1.dcm
                 -slice2.dcm
                 - ...
         - Paciente 2
                 - slice1.dcm
                 -slice2.dcm
                 - ...

"""
PATH ='E:/RubenSilva/PericardiumSegmentation/Dataset/OSIC/train' 
PathNames= 'E:/RubenSilva/PericardiumSegmentation/Dataset/OSIC/Convex_Mask'
list_patients=sorted(os.listdir(PathNames))

nu=0
nu1=0
patients=[]
patients_prob=[]

"""Copy entire fold"""
# for patient in list_patients:
#     patients.append(patient)
#     n=0
    
#     p_fold=os.path.join(PATH,patient)
#     #Pasta onde queres guardar os pacientes
#     dicom_img_path="C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/DCM"
#     dicom_patient=os.path.join(dicom_img_path,patient)
#     shutil.copytree(p_fold,dicom_patient) 
#     nu+=1
#     print(patient, nu)       

"""Move file"""     
# for file_name in list_patients:
    
#     #Pasta onde queres guardar os nrrd
#     path_move="E:/RubenSilva/PericardiumSegmentation/Dataset/OSIC/DCM"
#     fold_to_move=os.path.join(PATH,file_name[0:-5])
#     try:
#         #shutil.move(fold_to_move,path_move, copy_function = shutil.copytree)
#         nu+=1
#         print(file_name, nu,fold_to_move)    
#     except: 
#         print('no file', file_name)  
#         continue             
    

"""Move fold"""

for patient in list_patients:
    patient=patient.upper()
    #Pasta onde queres guardar os nrrd
    path_move="E:/RubenSilva/PericardiumSegmentation/Dataset/OSIC/DCM"
    fold_to_move=os.path.join(PATH,patient)
    try:
        shutil.move(fold_to_move,path_move, copy_function = shutil.copytree)
        nu+=1
        print(patient, nu,fold_to_move)    
    except: 
        print('no file', patient)  
        continue     




