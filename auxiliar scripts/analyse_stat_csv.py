# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:55:26 2024

@author: RubenSilva
"""

import pandas as pd
import numpy as np
import os
import csv
import openpyxl
# Replace 'file_path.csv' with the path to your CSV file
file_path = 'C:/Users/RubenSilva/Desktop/CardiacCT_peri/train.csv'
osic_path='E:/OSIC/Data'
# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)


list_patients=os.listdir(osic_path)

new_df = pd.DataFrame(columns=df.columns)

# dic={}
# for patient in list_patients:
#     csv_patient=df[df['Patient']==patient]
#     try:
#         csv_patient=csv_patient.iloc[[0]]
#     except:
#         pass
    
#     if patient in new_df['Patient']:
        
#         pass
#     else:
        
#         new_df = pd.concat([new_df, csv_patient], ignore_index=True)

new_df['Sex'].value_counts()        
new_df['SmokingStatus'].value_counts()