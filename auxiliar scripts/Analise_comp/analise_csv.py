# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 12:36:51 2023

@author: RubenSilva
"""
import os
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

import numpy as np 
from matplotlib import pyplot as plt
import numpy as np
import glob
import cv2
import time
import pandas as pd
"Import CSV with dicom and masks informations"

#path1='X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/classification_models/predict/2D_CNN/BCE/Cardiac_fat_tif/L0_case1_tif_calc_augm/th_0.517566/Peri_segm/NRRD'
path1='E:/RubenSilva/PericardiumSegmentation/Results/EPICHEART/Mon_Jul_15_17_18_23_2024/WithPP'
path2='E:/RubenSilva/PericardiumSegmentation/Results/EPICHEART/Mon_Jul_22_19_40_20_2024/WithPP'

case1= pd.read_excel(os.path.join(path1,'Dice.xlsx'))
case2 = pd.read_excel(os.path.join(path2,'Dice.xlsx'))

"Análise Dice"

case1_dcs,case2_dsc=case1['Dice'],case2['Dice']

#name1=path1.split('/')[-4]+'_slc_2.5d'
name1='NoPadding_Abd'
name2='NoPadding_All'

path_save_csv='E:/RubenSilva/PericardiumSegmentation/Results/EPICHEART/Comparations'
print(name1,name2,path_save_csv)
if not os.path.exists(path_save_csv):                         
     # Create a new directory because it does not exist 
     os.makedirs(path_save_csv)
os.chdir(path_save_csv) 

f= open('report'+name1+"_vs_"+name2+".txt","w+")

limit1=0.05
patients_to_see_dcs=[]
f.write("Dice analysis with limit: "+str(limit1)+'\r\n')
for i,dice in enumerate(case1_dcs):
    patient=case1['Patient'][i]
    p_index = case2.index[case2['Patient'] == patient][0]
    case2_dsc=case2['Dice'][p_index]
    dif=abs(dice-case2_dsc)
    print(patient,case2_dsc)
    if dif>limit1:
       patients_to_see_dcs.append(case1['Patient'][i]) 
       f.write("Patient: "+str(case1['Patient'][i])+', ')
       f.write(name1+': ' +str(dice)+', '+name2+' :'+ str(case2_dsc) +'\n')    
              
   
# "Análise jaccard"    

# case1_jcc,case2_jcc=case1['jaccard'],case2['jaccard']
# f.write('\r\n')
# patients_to_see_jcc=[]
# f.write("Jaccard analysis with limit: "+str(limit1)+'\r\n')
# for i,jcc in enumerate(case1_jcc):
#     patient=case1['Patient'][i]
#     p_index = case2.index[case2['Patient'] == patient][0]
    
#     case2_jcc=case2['jaccard'][p_index]
#     dif=abs(jcc-case2_jcc)
    
#     if dif>limit1:
#        patients_to_see_jcc.append(case1['Patient'][i])
#        f.write("Patient: "+str(case1['Patient'][i])+', ') 
#        f.write(name1+': ' +str(jcc)+', '+name2+' :'+ str(case2_jcc) +'\n')        
#            # f.write("Date end: "+(local_time_end)+'\n')

# "Análise hd"    
# limithd=50

# case1_hd,case2_hd=case1['hd'],case2['hd']
# f.write('\r\n')
# patients_to_see_hd=[]
# f.write("HD analysis with limit: "+str(limithd)+'\r\n')
# for i,hd in enumerate(case1_hd):
#     patient=case1['Patient'][i]
#     p_index = case2.index[case2['Patient'] == patient][0]
    
#     case2_hd=case2['hd'][p_index]
      
#     dif=abs(hd-case2_hd)
    
#     if dif>limithd:
#        patients_to_see_hd.append(case1['Patient'][i])
#        f.write("Patient: "+str(case1['Patient'][i])+', ')
#        f.write(name1+': ' +str(hd)+', '+name2+' :'+ str(case2_hd) +'\n')  
# "Análise mad"    

# limitmad=0.4

# case1_mad,case2_mad=case1['mad'],case2['mad']
# f.write('\r\n')
# patients_to_see_mad=[]
# f.write("MAD analysis with limit: "+str(limitmad)+'\r\n')
# for i,mad in enumerate(case1_mad):
#     patient=case1['Patient'][i]
#     p_index = case2.index[case2['Patient'] == patient][0]
    
#     case2_mad=case2['mad'][p_index]
    
     
#     dif=abs(mad-case2_mad)
     
    
#     if dif>limitmad:
#         patients_to_see_mad.append(case1['Patient'][i])   
#         f.write("Patient: "+str(case1['Patient'][i])+', ')
#         f.write(name1+': ' +str(mad)+', '+name2+' :'+ str(case2_mad) +'\n')  

f.close()    