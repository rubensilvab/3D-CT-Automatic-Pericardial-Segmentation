# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:24:55 2024

@author: RubenSilva
"""

import os
import shutil

# Function to read patient IDs from a text file
def read_patient_ids(file_path):
    with open(file_path, 'r') as file:
        patient_ids = [line.strip() for line in file.readlines()]
    return patient_ids

# Function to remove patient folders based on IDs
def remove_patients(folder_path, patient_ids):
    for patient_id in patient_ids:
        patient_folder = os.path.join(folder_path, patient_id)
        if os.path.exists(patient_folder):
            print(f"Removing patient folder: {patient_folder}")
            shutil. rmtree(patient_folder)

os.chdir('C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE') 

# Define paths
text_file_path = "patient_seg_er.txt"  # Path to the text file containing patient IDs
folder_path = "C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/PERI_segm_tif"         # Path to the folder containing patient data

# Read patient IDs from the text file
patient_ids_rw = read_patient_ids(text_file_path)
patient_ids=patient_ids_rw[2:]
# Remove patient folders based on IDs
#remove_patients(folder_path, patient_ids)
