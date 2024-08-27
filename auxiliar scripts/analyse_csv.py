# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:26:52 2024

@author: RubenSilva
"""

# read CSV, count the pericardium for each pacient.

import pandas as pd
import numpy as np
import os
import csv
import openpyxl
# Replace 'file_path.csv' with the path to your CSV file
file_path = 'CHVNGE/perict5.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

list_patients=np.unique(df['Patient'])

dic={}
for patient in list_patients:
    csv_patient=df[df['Patient']==patient]
    print(patient,'slices com pericardio:',np.sum(csv_patient['Label']))
    dic[patient]=np.sum(csv_patient['Label'])


csv_file_path=os.path.join('CHVNGE','npericardium.xlsx')

    
#filename = str(name)+'.xlsx'
# Create a new workbook object
book = openpyxl.Workbook()
# Select the worksheet to add data to
sheet = book.active
# Add a header row to the worksheet
sheet.append( ['Patient', 'Npericardium'])   
for key, value in dic.items():
       
       sheet.append([key, value])    

# Save the workbook to a file
book.save(csv_file_path)

    