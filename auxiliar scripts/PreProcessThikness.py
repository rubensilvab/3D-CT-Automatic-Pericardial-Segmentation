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

dcm_info= pd.read_excel(os.path.join('E:/RubenSilva/PericardiumSegmentation/Dataset/OSIC','dcmInfo.xlsx'))

patients_th_15=dcm_info[dcm_info['Slice Thickness']<3]

path_dcm='E:/RubenSilva/PericardiumSegmentation/Dataset/OSIC/DCM'
path_save='E:/RubenSilva/PericardiumSegmentation/Dataset/OSIC/DCM_3mm'
path_dcm_move='E:/RubenSilva/PericardiumSegmentation/Dataset/OSIC/DCM_ORGN_Less3mm'

path_nrrd='E:/RubenSilva/PericardiumSegmentation/Dataset/OSIC/Peri_segm'
path_nrrd_save='E:/RubenSilva/PericardiumSegmentation/Dataset/OSIC/PERI_segm_3mm'
path_nrrd_move='E:/RubenSilva/PericardiumSegmentation/Dataset/OSIC/PERI_segm_Less3mm'

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

patients=patients_th_15['Patient']
#patients=['ID00322637202284842245491']
for patient in patients:
    patient=str(patient)
    line_patient=patients_th_15[patients_th_15['Patient']==patient]
    thik=line_patient['Slice Thickness']
    #print(patient,thik)
    ratio=int(3/thik)
    
    # files=glob.glob(os.path.join(path_dcm,str(patient),'*'))
    # files=OrderSlices(files)
    
    # s=0
    # for i in range(len(files)):
        
    #     try:
    #         file_to_copy=files[s] 
    #         path_save_p=os.path.join(path_save,str(patient))
    #         isExist = os.path.exists(path_save_p)
    #         if not isExist:                         
    #             # Create a new directory because it does not exist 
    #             os.makedirs(path_save_p)
            
    #         shutil.copy(file_to_copy,path_save_p)
    #         s+=ratio
            
            
    #     except:
    #         print('acabou:',patient,'slice',s,'n slices:',i, 'new_thik:',float(ratio*thik))
    #         break
        
    file_nrrd_path=os.path.join(path_nrrd,str(patient)+".nrrd")   
    file_nrrd,header=read_nrrd(file_nrrd_path)
    
    nrrd_list=[]
    
    s=0
    if ratio!=1:
        for i in range(file_nrrd.shape[0]):
              
            try:
                slice_n=file_nrrd[s] 
                nrrd_list.append(np.array(slice_n))
                #print(slice_n.shape)
                s+=ratio
               
               
            except Exception as error: 
                #print('Ocorreu um erro', error, patient) 
    
                print('acabou:',patient,'slice',s,'n slices:',i)
                nrrd_list=np.array(nrrd_list)
                header['space directions'][2][2]=float(ratio*thik)
                header['space directions']=abs(header['space directions'])
                
                isExist = os.path.exists(path_nrrd_save)
                if not isExist:                         
                    # Create a new directory because it does not exist 
                    os.makedirs(path_nrrd_save)
                os.chdir(path_nrrd_save) 
                 
                nrrd.write(str(patient)+'.nrrd', reshape_nrrd(nrrd_list),header=header)
                break
        
        shutil.move(file_nrrd_path,path_nrrd_move)
        #shutil.move(os.path.join(path_dcm,str(patient)),os.path.join(path_dcm_move,str(patient)))
    else:
        shutil.copy(file_nrrd_path,path_nrrd_move)
        shutil.copy(file_nrrd_path,path_nrrd_save)
        print(patient,'has ratio=',ratio,' change nothing')
        
    #ID00322637202284842245491 --ver este