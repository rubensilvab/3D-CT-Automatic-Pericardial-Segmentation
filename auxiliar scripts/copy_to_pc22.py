# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 18:34:22 2024

@author: RubenSilva
"""

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

def ConvertNRRD2Int8(nrrd_file):
    
    readdata, header = nrrd.read(nrrd_file)
    #print(header)
    nrrd_file=np.array(readdata)
    nrrd_file=nrrd_file.astype(np.uint8) 
    
    return nrrd_file,header

nu=0
nu1=0
patients=[]

"""Where the manual and pred are in separte folds"""
# PATH_manual ='C:/Users/RubenSilva/Desktop/CardiacCT_peri/CardiacFat/Peri_segm'
# Path_pred= 'E:/RubenSilva/PericardiumSegmentation/Results/CardiacFat/Mon_Jul_22_19_40_20_2024/WithPP'
# list_patients=sorted(os.listdir(Path_pred))


# for patient in list_patients:
    
#     #Pasta onde queres guardar os nrrd
#     path_move="//10.227.103.133/ChestXRay/RubenSilva/3DRUI_cfat_Augm_PP_new"
#     path_move_patient=os.path.join(path_move,patient)
    
#     isExist = os.path.exists(path_move_patient)
#     if not isExist:                         
#         # Create a new directory because it does not exist 
#         os.makedirs(path_move_patient)
    
#     file_nrrd_pred=os.path.join(Path_pred,patient,'NRRD',patient+'_pred.nrrd')
#     file_nrrd_manual=os.path.join(PATH_manual,patient+'.nrrd')
    
#     try:
#         shutil.copy(file_nrrd_pred,path_move_patient)
        
#         nrrd_manual, header=ConvertNRRD2Int8(file_nrrd_manual)
#         manual_name=os.path.join(path_move_patient,patient+'_manual.nrrd')
#         nrrd.write(manual_name, nrrd_manual,header=header)
        
#         #old_name=os.path.join(path_move_patient,patient+'.nrrd')
#         #new_name=os.path.join(path_move_patient,patient+'_manual.nrrd')
        
#         #os.rename(old_name, new_name)
#         nu+=1
#         print(file_nrrd_manual, nu) 
        
#     except Exception as error: 
#         print('Ocorreu um erro', error, patient) 
        
#         continue

"""Normal case"""

Path_pred= 'E:/RubenSilva/PericardiumSegmentation/Results/CardiacFat/3DnnUnet'
list_patients=sorted(os.listdir(Path_pred))


for patient in list_patients:
    
    #Pasta onde queres guardar os nrrd
    path_move="//10.227.103.133/ChestXRay/RubenSilva/3DnnUnet_cft"
    path_move_patient=os.path.join(path_move,patient)
    
    isExist = os.path.exists(path_move_patient)
    if not isExist:                         
        # Create a new directory because it does not exist 
        os.makedirs(path_move_patient)
    
    file_nrrd_pred=os.path.join(Path_pred,patient,patient+'_pred.nrrd')
    file_nrrd_manual=os.path.join(Path_pred,patient,patient+'_manual.nrrd')
    
    try:
        
        nrrd_manual, header_manual=ConvertNRRD2Int8(file_nrrd_manual)
        nrrd_pred, header_pred= ConvertNRRD2Int8(file_nrrd_pred)
        
        path_manual=os.path.join(path_move_patient,patient+'_manual.nrrd')
        path_pred=os.path.join(path_move_patient,patient+'_pred.nrrd')
        
        nrrd.write(path_manual, nrrd_manual,header=header_manual)
        nrrd.write(path_pred, nrrd_pred,header=header_pred)
        
        
        nu+=1
        print(file_nrrd_manual, nu) 
        
    except Exception as error: 
        print('Ocorreu um erro', error, patient) 
        
        continue



