# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:43:40 2024

@author: RubenSilva
"""

import pydicom 
import glob


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

import numpy as np
import os
path_dcm='E:/RubenSilva/PericardiumSegmentation/Dataset/OSIC/DCM/'

list_patients=os.listdir(path_dcm)
list_patients=['ID00015637202177877247924']

"""ID00015637202177877247924 [ 0.8  1.2 21.6]
There is more thickness ID00068637202190879923934 [3. 6.]
There is more thickness ID00186637202242472088675 [0.7 1.4]
There is more thickness ID00381637202299644114027 [ 0.5  1.   1.5  2.5  4.5  5.   6.5  8.5 11.5 15.5 16. ]
There is more thickness ID00414637202310318891556 [1. 2.]"""

for patient in list_patients:
    
    files_dcm=sorted(glob.glob(path_dcm+str(patient)+'/*'))
    files_dcm=OrderSlices(files_dcm)
    #print(files_dcm[0])
    
    thk=[]
    
    for i in range(len(files_dcm)-1):
    # manufacturer, pixel spacinf, thikness
        dicom_data1 = pydicom.read_file(files_dcm[i])
        dicom_data2 = pydicom.read_file(files_dcm[i+1])

        real=RealThickness(dicom_data1,dicom_data2)
        real=round(real,1)
        thk.append(real)
        print(i,real)
    
    if len(np.unique(thk))>1:
        print('There is more thickness', patient, np.unique(thk))
