# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:30:32 2024

@author: RubenSilva
"""

import os
import nrrd
import cv2
import numpy as np
import matplotlib.pyplot as plt

path_nrrd='C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/PERI_segm'
path_tif='C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/PERI_segm_tif'


def reshape_nrrd(nrrd):
  
  nrrd=np.array(nrrd)
  nrrd=np.transpose(nrrd,(2,1,0))
  nrrd=nrrd[::-1]  
  return nrrd  


list_patients=sorted(os.listdir(path_nrrd))
patients_prob=[]
n=0
for patient in list_patients:
    
    patient=patient[:-5]
    PATH_nrd =os.path.join(path_nrrd, patient+'.nrrd') 
    try:
        # Read the data back from file
        n=n+1
        readdata, header = nrrd.read(PATH_nrd)
        new=reshape_nrrd(readdata)
        new=(new*65535).astype(np.uint16)
        # Plot the image
        #plt.imshow(new[39,:,:], cmap='gray')  
        #plt.show()
        for i in range(new.shape[0]):
            
            # save the  tif slice 
            path=os.path.join(path_tif,str(patient))
            isExist = os.path.exists(path)
            #print(path_to_cpy,isExist)
            if not isExist:                         
                # Create a new directory because it does not exist 
                os.makedirs(path)
            os.chdir(path) 
            order = '{:03d}'.format(i+1)
            
            cv2.imwrite(str(patient)+'_'+str(order)+'.tif', new[i,:,:].reshape(512,512,1))
            img_read=cv2.imread(str(patient)+'_'+str(order)+'.tif',flags=cv2.IMREAD_ANYDEPTH)
            #plt.imshow(img_read, cmap='gray')  
            #plt.show()
            print(patient,order, n)
    except:
        print("error , dont exist segm_nrd,patient :",patient)
        patients_prob.append(patient)
        #pass  