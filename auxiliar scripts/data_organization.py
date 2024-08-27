# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 21:50:00 2023

@author: RubenSilva
"""



import nrrd
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pydicom 
import numpy as np
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
import os
import glob
#from tensorflow.keras.preprocessing.image import load_img
#from tensorflow.keras.preprocessing.image import save_img
#from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import shutil
import openpyxl

"""Functions to process Dicom files"""

def window_image(img, window_center,window_width, intercept, slope,raw,rescale):
    img = (img*slope +intercept) #for translation adjustments given in the dicom file. 
    
    if raw==False:
        img_min = window_center - window_width//2 #minimum HU level
        img_max = window_center + window_width//2 #maximum HU level
        img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level
        img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level
        if rescale: 
            img = (img - img_min) / (img_max - img_min)*65535 # Para 16 bit
            #img = (img - img_min) / (img_max - img_min)*255.0 # Para 8 bit
            img=img.astype(np.uint16)
    return img

def reconvert_HU(array,window_center,window_width,L=0,W=2000,rescale=True):
    
    img_min = L - W//2 #minimum HU level
    img_max =L + W//2 #maximum HU level
    
    reconvert_img=(array/65535)*(img_max - img_min) +  img_min # Reconvertido para HU
    
    
    new_img_min = window_center - window_width//2 #minimum HU level, que pretendemos truncar
    new_img_max =window_center + window_width//2 #maximum HU level, que pretendemos truncar
    
    reconvert_img[reconvert_img<new_img_min] = new_img_min #set img_min for all HU levels less than minimum HU level
    reconvert_img[reconvert_img>new_img_max] = new_img_max #set img_max for all HU levels higher than maximum HU level
    
    if rescale: 
        reconvert_img = (reconvert_img - new_img_min) / (new_img_max - new_img_min)*65535 # Para 16 bit
        #reconvert_img = (reconvert_img - new_img_min) / (new_img_max - new_img_min)*255 # Para 8 bit
        reconvert_img=reconvert_img.astype(np.uint16)
     
        
    return reconvert_img
    
def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue: return int(x[0])
    else: return int(x)
    

  
def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def window_level(dicom_path,L=0,W=2000,raw=False,rescale=True):
  #Default: [-1000,1000] 
  
  data = pydicom.read_file(dicom_path)
  window_center , window_width, intercept, slope = get_windowing(data)
  img = pydicom.read_file(dicom_path).pixel_array
  img2 = window_image(img, L, W, intercept, slope,raw,rescale)   
  return img2    


def RealThickness(dcm1,dcm2):
    th1=dcm1.ImagePositionPatient[-1]
    th2=dcm2.ImagePositionPatient[-1]
    realthk=abs(th1-th2)
    
    return realthk
    

"Function to gather important information to CSV"    

def create_ExcelRowInfo(path,patient,name,dicom_path,dicom_path2,nslices):
    
    # manufacturer, pixel spacinf, thikness
    dicom_data = pydicom.read_file(dicom_path)
    dicom_data2=pydicom.read_file(dicom_path2)
    
    patient_id = dicom_data.PatientID
    patient_age = dicom_data.PatientAge
    patient_sex = dicom_data.PatientSex
    
    # Access manufacturer information
    manufacturer = dicom_data.Manufacturer 
    # Access pixel spacing information
    pixel_spacing = dicom_data.PixelSpacing
    # Access slice thickness information
    slice_thickness = RealThickness(dicom_data,dicom_data2) 
    image_dimensions = dicom_data.Rows, dicom_data.Columns
    
    # Study Information
    study_date = dicom_data.StudyDate
    study_description = dicom_data.StudyDescription
    
    # Series Information
    series_description = dicom_data.SeriesDescription
    series_number = dicom_data.SeriesNumber
    
    # Equipment Information
    manufacturer = dicom_data.Manufacturer
    manufacturer_model_name = dicom_data.ManufacturerModelName
    
    # Image Acquisition Parameters
    exposure_time = dicom_data.ExposureTime
    tube_current = dicom_data.XRayTubeCurrent
    tube_voltage = dicom_data.KVP

    
    
    
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

"""Funcions to adjust nrrd"""

def reshape_nrrd(nrrd):
  
  nrrd=np.array(nrrd)
  nrrd=np.transpose(nrrd,(2,1,0))
  nrrd=nrrd[::-1]  
  return nrrd  



"""
- Pasta principal(teste)
         - Paciente 1
                 - slice1.dcm
                 -slice2.dcm
                 - ...
         - Paciente 2
                 - slice1.dcm
                 -slice2.dcm
                 - ...

"""
PATH ='E:/CHVNGE_CT/Abdominal/Data_CCT/' 
PathNames= 'C:/Users/RubenSilva/Desktop/Segmentations_InesSousa'
list_patients=sorted(os.listdir(PathNames))

nu=0
nu1=0
patients=[]
patients_prob=[]

for patient in list_patients:
    patients.append(patient)
    n=0
    files=sorted(glob.glob(PATH+patient+'/*'))
    try:
        create_ExcelRowInfo('C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE',patient,'dcmInfo',files[0],files[1],len(files))
        n=n+1
        #print(patient,n,files[0])
    except:
        print('Paciente ', patient, 'deu problemas' )
        patients_prob.append(patient)
        pass
    
    for file in files:
            try: 
                n=n+1
                           
                "Apenas guardar imagem com -1000 a 1000 HU e 16 bit"
                
                #Pasta onde queres guardar as imagens .tif
                dicom_img_path="C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/DICOM"
                 
                path=os.path.join(dicom_img_path,str(patient))
                isExist = os.path.exists(path)
                if not isExist:                         
                    # Create a new directory because it does not exist 
                    os.makedirs(path)
                os.chdir(path) 
                 
                img=window_level(file)
                
                #img_dicom=window_level(file,raw=True)
                order = '{:03d}'.format(n)
                cv2.imwrite(str(patient)+'_'+str(order)+'.tif', img)
                img_read=cv2.imread(str(patient)+'_'+str(order)+'.tif',flags=cv2.IMREAD_ANYDEPTH)
                 
                nu=nu+1
                
                print(file,"dcm",n,"total:",nu)
                 
                
                # "Aplicando window level- só para fins de teste"
                # #Pasta onde vao ser guardadas estas imagens de teste
                
                # dicom_img_path="X:/Ruben/TESE/Data/hospital_gaia/imgs_tif/data1/WL"
             
                # path=os.path.join(dicom_img_path,str(patient))
                # isExist = os.path.exists(path)
                # if not isExist:                         
                # # Create a new directory because it does not exist 
                #   os.makedirs(path)
                # os.chdir(path) 
             
                # reconvert_img=reconvert_HU(img_read,50,350,rescale=True)
                # cv2.imwrite(str(patient)+'_'+str(n)+'.tif', reconvert_img)
                # img_read_reconvert=cv2.imread(str(patient)+'_'+str(n)+'.tif',flags=cv2.IMREAD_ANYDEPTH)
                
                # #plt.imshow(reconvert_img,cmap="gray")
                # #plt.show()
                # nu=nu+1
                # print(file,"dcm",n,"total:",nu)
             
    
                
            except:
                print("error ")
                patients_prob.append(patient)
                pass
    # try:
    #   file_nrrd =os.path.join('E:/CHVNGE_CT/Abdominal/Segms_CCT/manual_unk', patient+".nrrd") 
    #   shutil.copy(file_nrrd,'C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/PERI_segm')
    #   nu=nu+1
    #   print('Nrrd segm for patient: ',patient,", total:",nu1)

    # except:
    #     print("error , dont exist segm_nrd,patient :",patient)
    #     patients_prob.append(patient)
    #     pass

    
print("FEITO")



