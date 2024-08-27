# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 16:12:15 2023

@author: RubenSilva
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:47:02 2023

@author: RubenSilva
"""
import os
import glob
import numpy as np

import shutil
import csv
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd

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


#path="X:/Ruben/TESE/New_training_Unet/all_data/OSIC_new"  
path1='E:/RubenSilva/PericardiumSegmentation/Results/EPICHEART/Mon_Jul_22_19_40_20_2024/WithPP'
path2='E:/RubenSilva/PericardiumSegmentation/Results/EPICHEART/Mon_Jul_15_17_18_23_2024/WithPP'


list_patients=[729973,
786735,
774098,
536217,
391063
]

name1='NoPaddingTrain_all'
name2='NoPaddingTrain_abd'

comp=False # se queremos comparar ou nao,se nao: copiamos o name1

if comp:
    
    name_path=name1+'_VS_'+name2
    path_to_copy="E:/RubenSilva/PericardiumSegmentation/Results/EPICHEART/Comparations/Exemplo_Dice"

else:
    name_path="piores"
    path_to_copy="E:/RubenSilva/PericardiumSegmentation/Results/EPICHEART/Comparations/NoPadding_Alldts/Exemplo_Dice"

path_1_csv= pd.read_excel(os.path.join(path1,'Dice.xlsx')).iloc[:, :5]
path_2_csv= pd.read_excel(os.path.join(path2,'Dice.xlsx')).iloc[:, :5]
   
for patient in list_patients:
    
    files_1=sort_specific(sorted(glob.glob(os.path.join(path1,str(patient),'Images2D'+'/*'))))
    files_2=sort_specific(sorted(glob.glob(os.path.join(path2,str(patient),'Images2D'+'/*'))))
    
    inf1=path_1_csv.loc[path_1_csv['Patient']==int(patient)]
    inf2=path_2_csv.loc[path_2_csv['Patient']==int(patient)]
    
    for sli in range(len(files_1)):
        
          
            path_to_copy_3=os.path.join(path_to_copy,name_path, str(patient))
            isExist = os.path.exists(path_to_copy_3)
            if not isExist:                         
              # Create a new directory because it does not exist 
              os.makedirs(path_to_copy_3)
              
            os.chdir(path_to_copy_3)
            # Create the subplot
            # Create the plot with a single row and two columns
            
            if comp:
                fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,6))
                image1 = plt.imread(files_1[sli])
                image2 = plt.imread(files_2[sli])
                
            else:
                fig=plt.figure(figsize=(12,6))
                image1 = plt.imread(files_1[sli])
            
            if comp:
                
                # Set the first subplot with the first image
                ax[0].imshow(image1)
                ax[0].axis('off')
                ax[0].text(0.2, 1.1, str(name1), fontsize=11, ha='center', va='bottom', transform=ax[0].transAxes)
                ax[0].text(0.6, 1.1, str(inf1), fontsize=7, ha='left', va='bottom', transform=ax[0].transAxes)
    
    
                # Set the first subplot with the first image
                ax[1].imshow(image2)
                ax[1].axis('off')
                ax[0].text(0.2, -0.1, str(name2), fontsize=11, ha='center', va='bottom', transform=ax[0].transAxes)
                ax[0].text(0.6, -0.1, str(inf2), fontsize=7, ha='left', va='bottom', transform=ax[0].transAxes)

                
            else:
                
                plt.imshow(image1)
                plt.axis('off')
                #plt.text(0.2, 1.1, str(name1), fontsize=11, ha='center', va='bottom')
                plt.text(8, 1.1, str(inf1), fontsize=10, ha='left', va='bottom')
           
            
               
            # Save the subplot as a JPEG image
            plt.savefig(str(patient)+'_'+str(sli)+'.jpg', dpi=300, bbox_inches='tight')
            print(files_1[sli])
            print('')