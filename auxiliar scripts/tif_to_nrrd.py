# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:56:05 2024

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


#path_nrrd='E:/RubenSilva/PericardiumSegmentation/Dataset/OSIC/nrrd_heart'
path_tif='C:/Users/RubenSilva/Desktop/CardiacCT_peri/CardiacFat/Convex_Mask'
path_nrrd_move='C:/Users/RubenSilva/Desktop/CardiacCT_peri/CardiacFat/Peri_segm'

def sort_specific(files):
  sorted_files=[]
  for file in files:
         order=file[-7:-3]
         if order[1]=='_':
             sorted_files.append(file)
  for file in files:
         order=file[-7:-3]
         if order[0]=="_":
             sorted_files.append(file)  
  for file in files:
         order=file[-8:-3]
         if order[0]=="_":
             sorted_files.append(file)  
  return sorted_files  

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

def RealThickness(dcm1,dcm2):
    th1=dcm1.ImagePositionPatient[-1]
    th2=dcm2.ImagePositionPatient[-1]
    realthk=abs(th1-th2)
    
    return realthk

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


path_dcm='C:/Users/RubenSilva/Desktop/CardiacCT_peri/CardiacFat/DCM/'

list_patients=os.listdir(path_tif)
# list_patients=[
# 'ID00032637202181710233084',
# 'ID00075637202198610425520',
# 'ID00089637202204675567570',
# 'ID00104637202208063407045',
# 'ID00123637202217151272140'] 

# """ID00015637202177877247924',
# 'ID00032637202181710233084',
# 'ID00075637202198610425520',
# 'ID00089637202204675567570',
# 'ID00104637202208063407045',
# 'ID00123637202217151272140',
# 'ID00170637202238079193844',
# 'ID00381637202299644114027',
# 'ID00407637202308788732304',"""

"""For the OSIC when we have a nrrd reference"""


# for patient in list_patients:
    
#     files_dcm=sorted(glob.glob(path_dcm+str(patient)+'/*'))
#     files_dcm=OrderSlices(files_dcm)
#     print(files_dcm[0])
    
#     # file_tif_path=sorted(glob.glob(os.path.join(path_tif,patient,'*')))
#     # file_tif_path=sort_specific(file_tif_path)
#     #file_nrrd_path=os.path.join(path_nrrd,patient+'_heart.nrrd')
#     file_nrrd,header=read_nrrd(file_nrrd_path)
#     array_p=[]
    
#     for i,sli in enumerate(file_tif_path):
#         img_read=cv2.imread(sli,flags=cv2.IMREAD_ANYDEPTH)
#         #data = pydicom.read_file(files_dcm[i])
#         #dcm=data.pixel_array
        
#         img_read=(img_read/255).astype(np.uint8)
#         array_p.append(img_read)
        
#         # plt.imshow(dcm, cmap="gray")
                
#         # alpha=0.5
        
#         # plt.imshow(img_read,cmap="gray",alpha=alpha)
#         # #plt.imshow(file_nrrd[i],cmap="gray",alpha=alpha)
    
#         # plt.show()
        
        
        
#     array_p=np.array(array_p).astype(np.uint8)
#     # print(len(file_nrrd)==len(files_dcm)==len(array_p),patient)
    
#     # print(header['space directions'][2][2])
#     # print(header['space directions'],patient)
    
#     print('saving nrrd...', patient)         
#     nrrd.write(os.path.join(path_nrrd_move,str(patient)+'.nrrd'), reshape_nrrd(array_p),header=header)

"""For the Cardiac Fat where no nrrd of reference is present"""      
      
for patient in list_patients:
    
    files_dcm=sorted(glob.glob(path_dcm+str(patient)+'/*'))
    files_dcm=OrderSlices(files_dcm)
    # print(files_dcm[0])
    
    file_tif_path=sorted(glob.glob(os.path.join(path_tif,patient,'*')))
    file_tif_path=sort_specific(file_tif_path)
    # file_nrrd_path=os.path.join(path_nrrd,patient+'.nrrd')
    # file_nrrd,header=read_nrrd(file_nrrd_path)
    array_p=[]
    
    for i,sli in enumerate(file_tif_path):
        img_read=cv2.imread(sli,flags=cv2.IMREAD_ANYDEPTH)
        data = pydicom.read_file(files_dcm[i])
        dcm=data.pixel_array
        
        img_read=(img_read/255).astype(np.uint8)
        array_p.append(img_read)
        
        #plt.imshow(dcm, cmap="gray")
                
        #alpha=0.5
        
        #plt.imshow(img_read,cmap="gray",alpha=alpha)
        #plt.imshow(file_nrrd[i],cmap="gray",alpha=alpha)
    
        #plt.show()
        
    array_p=np.array(array_p).astype(np.uint8)
    print(len(files_dcm)==len(array_p),patient)
    
    # Extract metadata from the first DICOM file
    first_dicom,scd_dicom = pydicom.read_file(files_dcm[0]),pydicom.read_file(files_dcm[1])
    spacing = [float(first_dicom.PixelSpacing[0]), float(first_dicom.PixelSpacing[1]), float(RealThickness(first_dicom,scd_dicom))]
    
    orientation = np.reshape(first_dicom.ImageOrientationPatient, (2, 3))
    direction = np.vstack([orientation, np.cross(orientation[0], orientation[1])])
    last_dicom=pydicom.read_file(files_dcm[-1])
    # Create NRRD header
    header = {
        'type': 'uint8',
        'dimension': 3,
        'sizes': array_p.shape,
        'space': 'left-posterior-superior',
        'space directions': np.diag(spacing),
        'kinds': ['domain', 'domain', 'domain'],
        'endian': 'little',
        'encoding': 'gzip',
        'space origin': last_dicom.ImagePositionPatient
    }
    
        
    print(header['space directions'][2][2])
    # print(header['space directions'],patient)
    
    # print('saving nrrd...', patient)         
    # nrrd.write(os.path.join(path_nrrd_move,str(patient)+'.nrrd'), reshape_nrrd(array_p),header=header)

      