import pandas as pd
import numpy as np
import os
import pydicom
import nrrd
from torch.utils.data import Dataset
from tqdm import tqdm

class TrainDataset(Dataset):

    def __init__(self, csv_file, img_dir, seg_dir, img_size=512):

        self.csv = csv_file
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.img_size = img_size
        
        self.patients = self.csv['patient'].to_numpy()

    def __len__(self):
        
        return len(self.patients)

    def __getitem__(self, idx):
        
        img = self.get_img(self.patients[idx])
        seg = self.get_seg(self.patients[idx])
        
        img = img[::-1]
        img = np.transpose(img, (2, 1, 0))
        
        seg = seg[::-1]
        seg = np.transpose(seg, (2, 1, 0))
        
        head = {'space directions': np.diag(self.pix_spacing),
                'space': 'left-posterior-superior'}
            
        return img, seg, head
    
    def get_img(self, patient):
        
        path = os.path.join(self.img_dir, patient)
        path = os.path.join(path, os.listdir(path)[0])
        
        files = os.listdir(path)
        n_slices = len(files)
        
        imgs = np.zeros((n_slices, self.img_size, self.img_size))
        zcoords = []
        
        for i, file in enumerate(files):
            
            img = pydicom.read_file(os.path.join(path, file))
            if i==0:
                self.pix_spacing = img.get("PixelSpacing") 
                thick = img.get('SliceThickness')
                self.pix_spacing.append(thick)
            zcoords.append(img.get('ImagePositionPatient')[2])
            img = img.pixel_array * img.RescaleSlope + img.RescaleIntercept
            imgs[i] = img
            
        order = [i for i in range(n_slices)]
        new_order = [i for z, i in sorted(zip(zcoords, order))][::-1] 

        imgs = imgs[new_order].astype(np.int16)
        
        return imgs
    
    def  get_seg(self, patient):
        
        path = os.path.join(self.seg_dir, patient + '_lad.nrrd')
        
        lad, _ = nrrd.read(path)
        lad = np.transpose(lad, (2,1,0))
        lad = lad[::-1]
        
        path = os.path.join(self.seg_dir, patient + '_lcx.nrrd')
        
        lcx, _ = nrrd.read(path)
        lcx = np.transpose(lcx, (2,1,0))
        lcx = lcx[::-1]*2
        
        path = os.path.join(self.seg_dir, patient + '_lm.nrrd')
        
        lm, _ = nrrd.read(path)
        lm = np.transpose(lm, (2,1,0))
        lm = lm[::-1]*3
        
        path = os.path.join(self.seg_dir, patient + '_rca.nrrd')
        
        rca, _ = nrrd.read(path)
        rca = np.transpose(rca, (2,1,0))
        rca = rca[::-1]*4
        
        seg = lad + lcx + lm + rca
        
        return seg.astype(np.uint8)
    
def get_sets(data):
    
    train_folds = [1, 2, 3]
    #train_folds = [4]
    #train_folds = [5]
    
    train_set = data[data['fold'].isin(train_folds)]
    
    return train_set
    
if __name__ == "__main__":
    
    img_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\patient\\'
    seg_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\calcium_multi\\'
    
    img_path = 'C:\\Users\\RuiSantos\\Desktop\\CT\\nnUNet_raw\\Dataset111_COCA\\imagesTr\\'
    seg_path = 'C:\\Users\\RuiSantos\\Desktop\\CT\\nnUNet_raw\\Dataset111_COCA\\labelsTr\\'
    
    folds = pd.read_csv('C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\folds.csv')
    
    train = get_sets(folds)
    dset = TrainDataset(train, img_dir, seg_dir)
        
    for idx in tqdm(range(len(train))):
            
        img, seg, head = dset.__getitem__(idx)
            
        nrrd.write(os.path.join(img_path, "COCA_{:03d}_0000.nrrd".format(idx)), img, head)
        nrrd.write(os.path.join(seg_path, "COCA_{:03d}.nrrd".format(idx)), seg, head)
        
#%%

import pandas as pd
import numpy as np
import os
import pydicom
import nrrd
from torch.utils.data import Dataset
from tqdm import tqdm

class TrainDataset(Dataset):

    def __init__(self, csv_file, img_dir, seg_dir, img_size=512):

        self.csv = csv_file
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.img_size = img_size
        
        self.patients = self.csv['patient'].to_numpy()

    def __len__(self):
        
        return len(self.patients)

    def __getitem__(self, idx):
        
        img = self.get_img(self.patients[idx])
        seg = self.get_seg(self.patients[idx])
        
        img = img[::-1]
        img = np.transpose(img, (2, 1, 0))
        
        seg = seg[::-1]
        seg = np.transpose(seg, (2, 1, 0))
        
        head = {'space directions': np.diag(self.pix_spacing),
                'space': 'left-posterior-superior'}
            
        return img, seg, head
    
    def get_img(self, patient):
        
        path = os.path.join(self.img_dir, patient)
        path = os.path.join(path, os.listdir(path)[0])
        
        files = os.listdir(path)
        n_slices = len(files)
        
        imgs = np.zeros((n_slices, self.img_size, self.img_size))
        zcoords = []
        
        for i, file in enumerate(files):
            
            img = pydicom.read_file(os.path.join(path, file))
            if i==0:
                self.pix_spacing = img.get("PixelSpacing") 
                thick = img.get('SliceThickness')
                self.pix_spacing.append(thick)
            zcoords.append(img.get('ImagePositionPatient')[2])
            img = img.pixel_array * img.RescaleSlope + img.RescaleIntercept
            imgs[i] = img
            
        order = [i for i in range(n_slices)]
        new_order = [i for z, i in sorted(zip(zcoords, order))][::-1] 

        imgs = imgs[new_order].astype(np.int16)
        
        return imgs
    
    def  get_seg(self, patient):
        
        path = os.path.join(self.seg_dir, patient + '_lad.nrrd')
        
        lad, _ = nrrd.read(path)
        lad = np.transpose(lad, (2,1,0))
        lad = lad[::-1]
        
        path = os.path.join(self.seg_dir, patient + '_lcx.nrrd')
        
        lcx, _ = nrrd.read(path)
        lcx = np.transpose(lcx, (2,1,0))
        lcx = lcx[::-1]*2
        
        path = os.path.join(self.seg_dir, patient + '_lm.nrrd')
        
        lm, _ = nrrd.read(path)
        lm = np.transpose(lm, (2,1,0))
        lm = lm[::-1]*3
        
        path = os.path.join(self.seg_dir, patient + '_rca.nrrd')
        
        rca, _ = nrrd.read(path)
        rca = np.transpose(rca, (2,1,0))
        rca = rca[::-1]*4
        
        seg = lad + lcx + lm + rca
        
        return seg.astype(np.uint8)
    
def get_sets(data):

    train_folds = [4]
    
    train_set = data[data['fold'].isin(train_folds)]
    
    return train_set
    
if __name__ == "__main__":
    
    img_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\patient\\'
    seg_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\calcium_multi\\'
    
    img_path = 'C:\\Users\\RuiSantos\\Desktop\\CT\\nnUNet_raw\\Dataset111_COCA\\imagesTr\\'
    seg_path = 'C:\\Users\\RuiSantos\\Desktop\\CT\\nnUNet_raw\\Dataset111_COCA\\labelsTr\\'
    
    folds = pd.read_csv('C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\folds.csv')
    
    train = get_sets(folds)
    dset = TrainDataset(train, img_dir, seg_dir)
        
    for idx in tqdm(range(280, 280 + len(train))):
            
        img, seg, head = dset.__getitem__(idx - 280)
            
        nrrd.write(os.path.join(img_path, "COCA_{:03d}_0000.nrrd".format(idx)), img, head)
        nrrd.write(os.path.join(seg_path, "COCA_{:03d}.nrrd".format(idx)), seg, head)
        