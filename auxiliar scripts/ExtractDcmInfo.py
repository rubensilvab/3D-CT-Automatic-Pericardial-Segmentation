# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 17:02:08 2024

@author: RubenSilva
"""

"Function to gather important information to CSV"    

import pydicom 
import numpy as np
import os
import openpyxl

def RealThickness(dcm1,dcm2):
    th1=dcm1.ImagePositionPatient[-1]
    th2=dcm2.ImagePositionPatient[-1]
    realthk=abs(th1-th2)
    
    return realthk

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

def create_ExcelRowInfo(path,patient,name,dicom_path,dicom_path2,nslices):
    
    # manufacturer, pixel spacinf, thikness
    dicom_data = pydicom.read_file(dicom_path)
    dicom_data2=pydicom.read_file(dicom_path2)
    
    # Patient Information
    patient_id = getattr(dicom_data, 'PatientID', 'Nan')
    patient_age = getattr(dicom_data, 'PatientAge', 'Nan')
    patient_sex = getattr(dicom_data, 'PatientSex', 'Nan')
    
    # Manufacturer Information
    manufacturer = getattr(dicom_data, 'Manufacturer', 'Nan')
    manufacturer_model_name = getattr(dicom_data, 'ManufacturerModelName', 'Nan')
    
    # Pixel Spacing Information
    pixel_spacing = getattr(dicom_data, 'PixelSpacing', 'Nan')

    # Access slice thickness information
    slice_thickness = RealThickness(dicom_data,dicom_data2) 
    # Image Dimensions
    image_dimensions = getattr(dicom_data, 'Rows', 'Nan'), getattr(dicom_data, 'Columns', 'Nan')
    
    # Study Information
    study_date = getattr(dicom_data, 'StudyDate', 'Nan')
    study_description = getattr(dicom_data, 'StudyDescription', 'Nan')
    
    # Series Information
    series_description = getattr(dicom_data, 'SeriesDescription', 'Nan')
    series_number = getattr(dicom_data, 'SeriesNumber', 'Nan')
    
    # Image Acquisition Parameters
    exposure_time = getattr(dicom_data, 'ExposureTime', 'Nan')
    tube_current = getattr(dicom_data, 'XRayTubeCurrent', 'Nan')
    tube_voltage = getattr(dicom_data, 'KVP', 'Nan')
    
    
    isExist = os.path.exists(path)
    if not isExist:                         
        # Create a new directory because it does not exist 
        os.makedirs(path)
        
    os.chdir(path) 
    
        
    filename = str(name)+'.xlsx'
    
    # Check if the file already exists
    if not os.path.isfile(filename):
        # Create a new workbook object
        book = openpyxl.Workbook()
    
        # Select the worksheet to add data to
        sheet = book.active
    
        # Add a header row to the worksheet
        sheet.append(['Patient','PatientID', 'Age','Sex','Manufacturer','Model Name','Exposure Time','Tube Current','Tube Voltage', 'Pixel Spacing', 'Slice Thickness','Number Slices','Dimensions','Study Date', 'Study Description', 'Series Description','Series Number' ])
    else:
        # Open an existing workbook
        book = openpyxl.load_workbook(filename)
    
        # Select the worksheet to add data to
        sheet = book.active
    
    # Append a new row of data to the worksheet
    sheet.append([patient, patient_id, patient_age, patient_sex,manufacturer,manufacturer_model_name,exposure_time,tube_current,tube_voltage,str(pixel_spacing),slice_thickness,nslices, str(image_dimensions),study_date,study_description,series_description,series_number])
    #print([patient, slic, label, pred])
    # Save the workbook to a file
    book.save(filename)
    
PATH ='C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/DCM/' 

list_patients=sorted(os.listdir(PATH))
#list_patients=[133988,]
nu=0
nu1=0
patients=[]
patients_prob=[]

import glob
for patient in list_patients:
    n=0
    files=sorted(glob.glob(PATH+str(patient)+'/*'))
    files=OrderSlices(files)
    # if (files[0].split('\\')[-1]!='1.dcm'):
    #     print(patient)
    try:
        create_ExcelRowInfo('C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE',patient,'dcmInfoPreP',files[0],files[1],len(files))
        n=n+1
        print(patient,n,files[0])
    except Exception as error: 
        print('Paciente ', patient, 'deu problemas:', error )
        patients_prob.append(patient)
        pass
    
    