# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 10:48:12 2024

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

"Extrair dice"

#Andominal
path_abd='E:/RubenSilva/PericardiumSegmentation/Results/Abdominal/Mon_Jul_22_19_40_20_2024/WithPP'
path_abd_info='C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE'
dice_abd=((pd.read_excel(os.path.join(path_abd,'Dice.xlsx'))).iloc[:,:5]).dropna()
slices_abd=pd.read_excel(os.path.join(path_abd_info,'dcmInfoPreP.xlsx'))

#EPI
# path_epi='E:/RubenSilva/PericardiumSegmentation/Results/EPICHEART/Mon_Jul_22_19_40_20_2024/WithPP'
# path_epi_info='C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/EPICHEART'
# dice_epi=((pd.read_excel(os.path.join(path_epi,'Dice.xlsx'))).iloc[:,:5]).dropna()
# slices_epi=pd.read_excel(os.path.join(path_epi_info,'dcmInfo.xlsx'))


"Extract Dice vs number of slices"
# dice_list=[]
# slices=[]

# for i in range(len(dice_epi)):
#     patient=dice_epi.loc[i,'Patient']
#     dice=dice_epi.loc[i,'Dice']
#     row=slices_epi[slices_epi['Patient']==patient]
#     n_slices=row['Number Slices'].values[0]
#     dice_list.append(dice)
#     slices.append(n_slices)
    
# """Add n slices to dice csv"""
# dice_epi['N Slices']=slices    

# path_save='E:/RubenSilva/PericardiumSegmentation/Results/EPICHEART/Mon_Jul_22_19_40_20_2024/Dice_VS_Slices'
# # Save the DataFrame to an Excel file
# dice_epi.to_excel(os.path.join(path_save,'DiceVsSlices.xlsx'), index=False)

# unique_slices=np.unique(slices)

# # Initialize SliPind with empty lists for each unique value in N_slices
# SliPind = [[] for _ in range(len(unique_slices))]

# for i in range(len(SliPind)):
#     for n in range(len(slices)):
#         if slices[n]==unique_slices[i]:
#             SliPind[i].append(dice_list[n])

# df = pd.DataFrame(SliPind)    
# # Insert the new column at the beginning (index 0)
# df.insert(0, 'Slices', unique_slices)
# # Save the DataFrame to an Excel file
# df.to_excel(os.path.join(path_save,'DiceVsSlices_bxp.xlsx'), index=False)

"Same for the px size"

"Extract Dice vs number of slices"
dice_list=[]
px_sizes=[]

for i in range(len(dice_abd)):
    patient=dice_abd.loc[i,'Patient']
    dice=dice_abd.loc[i,'Dice']
    row=slices_abd[slices_abd['Patient']==patient]
    pxs=row.iloc[0,-1]
    dice_list.append(dice)
    px_sizes.append(round(pxs,2))
    
"""Add n slices to dice csv"""
dice_abd['Px size']=px_sizes    

path_save='E:/RubenSilva/PericardiumSegmentation/Results/Abdominal/Mon_Jul_22_19_40_20_2024/Dice_Vs_Slices'
# Save the DataFrame to an Excel file
dice_abd.to_excel(os.path.join(path_save,'DiceVsPxSize.xlsx'), index=False)

unique_sizes=np.unique(px_sizes)

# Initialize SliPind with empty lists for each unique value in N_slices
SliPind = [[] for _ in range(len(unique_sizes))]

for i in range(len(SliPind)):
    for n in range(len(px_sizes)):
        if px_sizes[n]==unique_sizes[i]:
            SliPind[i].append(dice_list[n])

df = pd.DataFrame(SliPind)    
# Insert the new column at the beginning (index 0)
df.insert(0, 'Px size', unique_sizes)
# Save the DataFrame to an Excel file
df.to_excel(os.path.join(path_save,'DiceVsPxSize_bxp.xlsx'), index=False)