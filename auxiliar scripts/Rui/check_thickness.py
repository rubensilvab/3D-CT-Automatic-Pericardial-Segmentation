import pydicom
import numpy as np
import os
import pandas as pd

class CustomDataGenerator():
    
    def __init__(self, path, files):

        self.path = path
        self.files = files
        self.origins = []
        
    def __len__(self):
       
        return len(self.files)
    
    def __getitem__(self, idx):
        
        path = os.path.join(self.path, self.files[idx])
        img = pydicom.read_file(path)

        self.origins.append(float(img.get('ImagePositionPatient')[2]))
    
    def get_batch(self, patient):
        
        for idx in range(self.__len__()):
            self.__getitem__(idx)
            
        self.origins = np.array(sorted(self.origins)[::-1])
        
        thickness = np.unique([round(value, 1) for value in np.roll(self.origins, 1) - self.origins if value>0])
        
        if len(thickness)==1 and thickness[0]==3:
            return patient
        else:
            print(patient, thickness)
            return None

#%%
    
if __name__=='__main__':
    
    path = 'datasets\\COCA\\patient\\'
    
    patients = [str(i) for i in range(451, 790) if str(i) in os.listdir(path)]
    
    lens = []
    
    for patient in patients:
        
        patient_path = os.path.join(path, patient)
        patient_path = os.path.join(patient_path, os.listdir(patient_path)[0])
        
        files = os.listdir(patient_path)
        lens.append([patient, len(files)])
        
        generator = CustomDataGenerator(patient_path, files)        
        generator.get_batch(patient)

    print(sorted(lens, key=lambda x: x[1]))
        
#%%
    
if __name__=='__main__':
    
    path = 'datasets\\GAIA\\Data_CCT\\'
    
    data = pd.read_csv('datasets\\GAIA\\data_processed.csv')
    data['nsc'] = data['nsc'].astype(str)
    names = list(data['nsc'])
    
    out_dir = 'datasets\\GAIA\\pericardium\\'
    
    new_names = []
    
    for patient in names:
        
        patient_path = os.path.join(path, patient)
        files = os.listdir(patient_path)
        
        generator = CustomDataGenerator(patient_path, files)
        
        name = generator.get_batch(patient)
        
        if name:
            new_names.append(name)
            
    data = data[data['nsc'].isin(new_names)]
    
    data.to_csv('datasets\\GAIA\\data_processed_v2.csv', index=False)