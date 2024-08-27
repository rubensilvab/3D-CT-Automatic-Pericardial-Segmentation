# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:52:36 2024

@author: RubenSilva
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:29:09 2024

@author: RubenSilva
"""

import os
import nrrd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pandas as pd
import glob
import pydicom


path_nrrd='C:/Users/RubenSilva/Desktop/CardiacCT_peri/CardiacFat/Peri_segm'
path_nrrd_save='C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/PERI_segm_3mm'
path_nrrd_move='C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/PERI_segm_1.5mm'

def reshape_nrrd(data):
  nrrd=data[::-1] 
  nrrd=np.transpose(nrrd,(2,1,0))  
 
  return nrrd    

def read_nrrd(file_mask):
    
    readdata, header = nrrd.read(file_mask)
    #print(header)
    mask=np.array(readdata)
    mask=np.transpose(mask,(2,1,0))
    mask=mask[::-1] 
    
    return mask,header

def RealThickness(dcm1,dcm2):
    th1=dcm1.ImagePositionPatient[-1]
    th2=dcm2.ImagePositionPatient[-1]
    realthk=abs(th1-th2)
    
    return realthk

def OrderSlices(files):
    
    n_slices=len(files)
    slices=[]
    zcoords=[]
    for i, file in enumerate(files):
        
        img = pydicom.read_file(file)
        zcoords.append(img.get('ImagePositionPatient')[2])
        slices.append(file)
        
    order = [i for i in range(n_slices)]
    new_order = [i for z, i in sorted(zip(zcoords, order))][::-1] 
    
    reordered_slices = [slices[i] for i in new_order]
    
    return reordered_slices

path_dcm='C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/DCM/'
path_nrrd='C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/Peri_segm'

for patient in os.listdir(path_dcm):
    
    files_dcm=sorted(glob.glob(path_dcm+str(patient)+'/*'))
    files_dcm=OrderSlices(files_dcm)
    
    file_nrrd_path=os.path.join(path_nrrd,patient+'.nrrd')   
    file_nrrd,header=read_nrrd(file_nrrd_path)
    
    # plt.imshow(file_nrrd[0,:,:], cmap='gray')  
    # plt.show()
    #print(header['space directions'][2][2])
    #print(header['space directions'],file_nrrd)
    
    first_dicom,scd_dicom = pydicom.read_file(files_dcm[0]),pydicom.read_file(files_dcm[1])
    spacing = [float(first_dicom.PixelSpacing[0]), float(first_dicom.PixelSpacing[1]), float(RealThickness(first_dicom,scd_dicom))]
    
    orientation = np.reshape(first_dicom.ImageOrientationPatient, (2, 3))
    direction = np.vstack([orientation, np.cross(orientation[0], orientation[1])])
    print(first_dicom.ImagePositionPatient)
    print('header:',header)         
    s
            #nrrd.write(str(patient)+'.nrrd', reshape_nrrd(nrrd_list),header=header)
            
    #shutil.move(file_nrrd_path,path_nrrd_move)
    #shutil.move(os.path.join(path_dcm,str(patient)),os.path.join(path_dcm_move,str(patient)))