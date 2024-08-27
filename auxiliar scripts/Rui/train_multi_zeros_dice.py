import pandas as pd
import numpy as np
import os
import pydicom
import random
import nrrd

import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from transforms_multi import TrainTransform, ValidTransform

class TrainDataset(Dataset):

    def __init__(self, csv_file, img_dir, per_dir, seg_dir, img_size=512, input_size=384, depth=40, transform=None):

        self.csv = csv_file
        self.img_dir = img_dir
        self.per_dir = per_dir
        self.seg_dir = seg_dir
        self.img_size = img_size
        self.input_size = input_size
        self.depth = depth
        self.transform = transform
        
        self.patients = self.csv['patient'].to_numpy()
        self.scores = self.csv['score'].to_numpy()

    def __len__(self):
        
        return len(self.patients)

    def __getitem__(self, idx):
        
        imgs = self.read_img(self.patients[idx])
        seg = self.read_seg(idx, imgs)
        
        coords = self.read_pericardium(self.patients[idx])
        
        imgs, seg = self.crop_depth(imgs, seg, coords)
        imgs, seg = self.crop_xy(imgs, seg, coords)

        if self.transform:
            imgs, seg = self.transform(imgs, seg)
        
        imgs = torch.unsqueeze(imgs, 0)
        
        imgs = imgs.float()
        seg = seg.float()
        
        background = (torch.sum(seg, dim=0, keepdim=True) == 0).float()
        seg = torch.cat((seg, background), dim=0)
            
        return imgs, seg
    
    def read_img(self, patient):
        
        path = os.path.join(self.img_dir, patient)
        path = os.path.join(path, os.listdir(path)[0])
        
        files = os.listdir(path)
        n_slices = len(files)
        
        imgs = np.zeros((n_slices, self.img_size, self.img_size))
        zcoords = []
        
        for i, file in enumerate(files):
            
            img = pydicom.read_file(os.path.join(path, file))
            zcoords.append(img.get('ImagePositionPatient')[2])
            img = img.pixel_array * img.RescaleSlope + img.RescaleIntercept
            img = self.clip_values(img)
            imgs[i] = img
            
        order = [i for i in range(n_slices)]
        new_order = [i for z, i in sorted(zip(zcoords, order))][::-1] 

        imgs = imgs[new_order]
        imgs = torch.from_numpy(imgs)
        
        return imgs
    
    def read_pericardium(self, patient):
        
        path = os.path.join(self.per_dir, patient, 'segm_rui.nrrd')
        
        per, _ = nrrd.read(path)
        per = np.transpose(per, (2,1,0))
        per = per[::-1].astype(np.uint8)
        
        z = np.nonzero(np.sum(per, axis=(1,2)))
        zmin, zmax = np.min(z), np.max(z)
        
        y = np.nonzero(np.sum(per, axis=(0,2)))
        ymin, ymax = np.min(y), np.max(y)
        
        x = np.nonzero(np.sum(per, axis=(0,1)))
        xmin, xmax = np.min(x), np.max(x)
        
        return [zmin, zmax, ymin, ymax, xmin, xmax]
    
    def  read_seg(self, idx, imgs):
        
        if self.scores[idx]==0:
            seg = torch.zeros((4, imgs.shape[0], imgs.shape[1], imgs.shape[2]))
            
        else:
            patient = self.patients[idx]
            path = os.path.join(self.seg_dir, patient + '_lad.nrrd')
            
            lad, _ = nrrd.read(path)
            lad = np.transpose(lad, (2,1,0))
            lad = lad[::-1]
            lad = torch.from_numpy(lad.copy())
            
            path = os.path.join(self.seg_dir, patient + '_lcx.nrrd')
            
            lcx, _ = nrrd.read(path)
            lcx = np.transpose(lcx, (2,1,0))
            lcx = lcx[::-1]
            lcx = torch.from_numpy(lcx.copy())
            
            path = os.path.join(self.seg_dir, patient + '_lm.nrrd')
            
            lm, _ = nrrd.read(path)
            lm = np.transpose(lm, (2,1,0))
            lm = lm[::-1]
            lm = torch.from_numpy(lm.copy())
            
            path = os.path.join(self.seg_dir, patient + '_rca.nrrd')
            
            rca, _ = nrrd.read(path)
            rca = np.transpose(rca, (2,1,0))
            rca = rca[::-1]
            rca = torch.from_numpy(rca.copy())
            
            seg = torch.stack((lad, lcx, lm, rca))
        
        return seg
    
    def crop_depth(self, imgs, seg, coords):
        
        n_heart = coords[1] - coords[0] + 1
        
        if n_heart > self.depth:
            diff = n_heart - self.depth
            start = random.randint(0, diff)
                
            imgs = imgs[coords[0] + start : coords[0] + start + self.depth]
            seg = seg[:, coords[0] + start : coords[0] + start + self.depth]
            
            return imgs, seg
            
        elif n_heart < self.depth:
            n_slices = len(imgs)
            
            if n_slices < self.depth: 
                imgs, seg = self.pad_depth(imgs, seg, n_slices)
                
                return imgs, seg
            
            elif n_slices > self.depth:
                diff = n_slices - self.depth
                start = random.randint(max(0, coords[1] - self.depth + 1), min(diff, coords[0]))
                        
                imgs = imgs[start : start + self.depth]
                seg = seg[:, start : start + self.depth]
                
                return imgs, seg
            
            else:  
                return imgs, seg
            
        else:
            imgs = imgs[coords[0] : coords[1] + 1]
            seg = seg[:, coords[0] : coords[1] + 1]
            
            return imgs, seg
        
    def crop_xy(self, imgs, seg, coords):
        
        height = coords[3] - coords[2] + 1
        width = coords[5] - coords[4] + 1 
        
        if height <= self.input_size and width <= self.input_size:
            diff_h = self.input_size - height
            diff_w = self.input_size - width
            
            start_h = random.randint(max(0, coords[2] - diff_h), min(coords[2], self.img_size - self.input_size))
            start_w = random.randint(max(0, coords[4] - diff_w), min(coords[4], self.img_size - self.input_size))
                        
            imgs = imgs[:, start_h : start_h + self.input_size, start_w : start_w + self.input_size]
            seg = seg[:, :, start_h : start_h + self.input_size, start_w : start_w + self.input_size]
            
        else:
            if height >= width:
                diff_w = height - width
                start_w = random.randint(max(0, coords[4] - diff_w), min(coords[4], self.img_size - height))
                
                imgs = imgs[:, :, start_w : start_w + height]
                seg = seg[:, :, :, start_w : start_w + height]
                
            else:
                diff_h = width - height
                start_h = random.randint(max(0, coords[2] - diff_h), min(coords[2], self.img_size - width))
                
                imgs = imgs[:, start_h : start_h + width, :]
                seg = seg[:, :, start_h : start_h + width, :]
                
        return imgs, seg

    def clip_values(self, img, low=-800, high=2000):
                
        img[img<low] = low
        img[img>high] = high
            
        return img
    
    def pad_depth(self, imgs, seg, n_slices):
                
        diff = self.depth - n_slices
    
        imgs_pad = torch.full((self.depth, self.img_size, self.img_size), fill_value=-1000)
        imgs_pad[diff//2 : diff//2 + n_slices] = imgs
                
        seg_pad = torch.zeros((seg.shape[0], self.depth, self.img_size, self.img_size))
        seg_pad[:, diff//2 : diff//2 + n_slices] = seg
                
        return imgs_pad, seg_pad
    
class ValidDataset(Dataset):

    def __init__(self, csv_file, img_dir, per_dir, seg_dir, img_size=512, input_size=384, depth=40, transform=None):

        self.csv = csv_file
        self.img_dir = img_dir
        self.per_dir = per_dir
        self.seg_dir = seg_dir
        self.img_size = img_size
        self.input_size = input_size
        self.depth = depth
        self.transform = transform
        
        self.patients = self.csv['patient'].to_numpy()
        self.scores = self.csv['score'].to_numpy()

    def __len__(self):
        
        return len(self.patients)

    def __getitem__(self, idx):
        
        imgs = self.read_img(self.patients[idx])
        seg = self.read_seg(idx, imgs)
        
        coords = self.read_pericardium(self.patients[idx])
        
        imgs, seg = self.crop_depth(imgs, seg, coords)
        imgs, seg = self.crop_xy(imgs, seg, coords)

        if self.transform:
            imgs, seg = self.transform(imgs, seg)
        
        imgs = torch.unsqueeze(imgs, 0)
            
        imgs = imgs.float()
        seg = seg.float()
        
        background = (torch.sum(seg, dim=0, keepdim=True) == 0).float()
        seg = torch.cat((seg, background), dim=0)
            
        return imgs, seg
    
    def read_img(self, patient):
        
        path = os.path.join(self.img_dir, patient)
        path = os.path.join(path, os.listdir(path)[0])
        
        files = os.listdir(path)
        n_slices = len(files)
        
        imgs = np.zeros((n_slices, self.img_size, self.img_size))
        zcoords = []
        
        for i, file in enumerate(files):
            
            img = pydicom.read_file(os.path.join(path, file))
            zcoords.append(img.get('ImagePositionPatient')[2])
            img = img.pixel_array * img.RescaleSlope + img.RescaleIntercept
            img = self.clip_values(img)
            imgs[i] = img
            
        order = [i for i in range(n_slices)]
        new_order = [i for z, i in sorted(zip(zcoords, order))][::-1] 

        imgs = imgs[new_order]
        imgs = torch.from_numpy(imgs)
        
        return imgs
    
    def read_pericardium(self, patient):
        
        path = os.path.join(self.per_dir, patient, 'segm_rui.nrrd')
        
        per, _ = nrrd.read(path)
        per = np.transpose(per, (2,1,0))
        per = per[::-1].astype(np.uint8)
        
        z = np.nonzero(np.sum(per, axis=(1,2)))
        zmin, zmax = np.min(z), np.max(z)
        
        y = np.nonzero(np.sum(per, axis=(0,2)))
        ymin, ymax = np.min(y), np.max(y)
        
        x = np.nonzero(np.sum(per, axis=(0,1)))
        xmin, xmax = np.min(x), np.max(x)
        
        return [zmin, zmax, ymin, ymax, xmin, xmax]
    
    def  read_seg(self, idx, imgs):
        
        if self.scores[idx]==0:
            seg = torch.zeros((4, imgs.shape[0], imgs.shape[1], imgs.shape[2]))
            
        else:
            patient = self.patients[idx]
            path = os.path.join(self.seg_dir, patient + '_lad.nrrd')
            
            lad, _ = nrrd.read(path)
            lad = np.transpose(lad, (2,1,0))
            lad = lad[::-1]
            lad = torch.from_numpy(lad.copy())
            
            path = os.path.join(self.seg_dir, patient + '_lcx.nrrd')
            
            lcx, _ = nrrd.read(path)
            lcx = np.transpose(lcx, (2,1,0))
            lcx = lcx[::-1]
            lcx = torch.from_numpy(lcx.copy())
            
            path = os.path.join(self.seg_dir, patient + '_lm.nrrd')
            
            lm, _ = nrrd.read(path)
            lm = np.transpose(lm, (2,1,0))
            lm = lm[::-1]
            lm = torch.from_numpy(lm.copy())
            
            path = os.path.join(self.seg_dir, patient + '_rca.nrrd')
            
            rca, _ = nrrd.read(path)
            rca = np.transpose(rca, (2,1,0))
            rca = rca[::-1]
            rca = torch.from_numpy(rca.copy())
            
            seg = torch.stack((lad, lcx, lm, rca))
        
        return seg
    
    def crop_depth(self, imgs, seg, coords):
        
        n_slices = len(imgs)
        
        if n_slices < self.depth:
            imgs, seg = self.pad_depth(imgs, seg, n_slices)
            
            return imgs, seg
        
        elif n_slices > self.depth: 
            center = (coords[0] + coords[1]) // 2
            start = min(max(0, center - self.depth // 2), n_slices - self.depth)
            
            imgs = imgs[start : start + self.depth]
            seg = seg[:, start : start + self.depth]
        
            return imgs, seg
        
        return imgs, seg
        
    def crop_xy(self, imgs, seg, coords):
        
        height = coords[3] - coords[2] + 1
        width = coords[5] - coords[4] + 1 
        
        if height < self.input_size and width < self.input_size:
            center_h = (coords[2] + coords[3]) // 2
            center_w = (coords[4] + coords[5]) // 2
            
            start_h = min(max(0, center_h - self.input_size // 2), self.img_size - self.input_size)
            start_w = min(max(0, center_w - self.input_size // 2), self.img_size - self.input_size)
                        
            imgs = imgs[:, start_h : start_h + self.input_size, start_w : start_w + self.input_size]
            seg = seg[:, :, start_h : start_h + self.input_size, start_w : start_w + self.input_size]
            
        elif height > self.input_size or width > self.input_size:
            if height > width:
                center_w = (coords[4] + coords[5]) // 2
                start_w = min(max(0, center_w - height // 2), self.img_size - height)
                            
                imgs = imgs[:, :, start_w : start_w + height]
                seg = seg[:, :, :, start_w : start_w + height]
                
            elif height < width:
                center_h = (coords[2] + coords[3]) // 2
                start_h = min(max(0, center_h - width // 2), self.img_size - width)
                            
                imgs = imgs[:, start_h : start_h + width, :]
                seg = seg[:, :, start_h : start_h + width, :]
                
        return imgs, seg
        
    def clip_values(self, img, low=-800, high=2000):
                
        img[img<low] = low
        img[img>high] = high
            
        return img
    
    def pad_depth(self, imgs, seg, n_slices):
                
        diff = self.depth - n_slices
    
        imgs_pad = torch.full((self.depth, self.img_size, self.img_size), fill_value=-1000)
        imgs_pad[diff//2 : diff//2 + n_slices] = imgs
                
        seg_pad = torch.zeros((seg.shape[0], self.depth, self.img_size, self.img_size))
        seg_pad[:, diff//2 : diff//2 + n_slices] = seg
                
        return imgs_pad, seg_pad
    
def get_sets(split, folds):
    
    train_folds = [[1,2,3], [2,3,4], [3,4,5], [4,5,1], [5,1,2]]
    valid_folds = [4,5,1,2,3]
    test_folds = [5,1,2,3,4]
    
    train_set = folds[folds['fold'].isin(train_folds[split])]
    valid_set = folds[folds['fold']==valid_folds[split]]
    test_set = folds[folds['fold']==test_folds[split]]
    
    return train_set, valid_set, test_set

def dice_loss(target, output, batch_size, rank, factor=0.2, out_channels=5):
    
    loss = torch.zeros(batch_size).to(rank)
    
    for i in range(batch_size):
        for j in range(out_channels):
            if torch.sum(target[i, j])>0:
        
                dice = 1 - 2 * torch.sum(target[i, j] * output[i, j]) / (torch.sum(target[i, j]**2) + torch.sum(output[i, j]**2) + 1e-7)
            
                loss[i] += factor * dice
        
    return loss.mean()
    
def train_model(params, model, train_dl, valid_dl, path, rank, world_size, run, split):
    
    if rank==0:  
        writer = SummaryWriter(log_dir='C:/Users/RuiSantos/Desktop/CT/runs/' + f"run_{run}_split_{split}")
    
    count = -1
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(params['n_epochs']):
        
        model.train()
        train_dl.sampler.set_epoch(epoch)
        
        if rank==0:
            epoch_loss = 0
            counter = 0
                
        for imgs, segs in train_dl:
                    
            imgs = imgs.to(rank)  
            segs = segs.to(rank)
            
            with torch.cuda.amp.autocast():
                preds = model(imgs)
    
                loss = dice_loss(segs, preds, params['batch_size'], rank)
                scaler.scale(loss).backward()
                    
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad(set_to_none=True)
                    
            if rank==0:
                batch_loss = [None for _ in range(world_size)]
                dist.gather_object(loss.item(), object_gather_list=batch_loss)
                
                epoch_loss += sum(batch_loss) / world_size
                counter += 1
                
            else:
                dist.gather_object(loss.item())
                
        if rank==0:
            writer.add_scalar('Train loss', epoch_loss/counter, epoch)

        with torch.no_grad():
            model.eval()
            
            if rank==0: 
                epoch_loss = 0
                counter = 0
            
            for imgs, segs in valid_dl:
                
                imgs = imgs.to(rank)
                segs = segs.to(rank)
                
                with torch.cuda.amp.autocast():
                    preds = model(imgs)
    
                    loss = dice_loss(segs, preds, params['batch_size'], rank)
                
                if rank==0:
                    batch_loss = [None for _ in range(world_size)]
                    dist.gather_object(loss.item(), object_gather_list=batch_loss)
                    
                    epoch_loss += sum(batch_loss) / world_size
                    counter += 1
                        
                else:
                    dist.gather_object(loss.item())
                    
            if rank==0:
                writer.add_scalar('Valid loss', epoch_loss/counter, epoch)
        
        model.train()
        
        if rank==0:
            torch.save(model.module.state_dict(), os.path.join(path, f"epoch_{epoch}.pth"))
            
            if epoch == 0:
                min_loss = epoch_loss
                
            if min_loss - epoch_loss > params['delta']:
                count = 0
                min_loss = epoch_loss
            else:
                count += 1

            dist.gather_object(count, dst=1)
            
        else:
            counts = [None for _ in range(world_size)]
            dist.gather_object(count, object_gather_list=counts, dst=1)
            count = max(counts)
                
        if count == params['patience']:
            if rank==0:
                writer.flush()
                writer.close()
            return None
          
    if rank==0:
        writer.flush()
        writer.close()
    return None

def main(rank, world_size, train_dataset, valid_dataset, params, path, run, split):
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, drop_last=True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_dl = DataLoader(train_dataset, params['batch_size'], sampler=train_sampler, drop_last=True) 
    valid_dl = DataLoader(valid_dataset, params['batch_size'], sampler=valid_sampler, drop_last=True)
                
    model = CAN3D().to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    train_model(params, model, train_dl, valid_dl, path, rank, world_size, run, split)

    dist.destroy_process_group()

from can3d_multi_nodil import CAN3D
    
if __name__ == "__main__":
    
    run = 84
    
    img_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\patient\\'
    per_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\pericardium\\'
    seg_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\calcium_multi\\'
    
    model_path = 'C:\\Users\\RuiSantos\\Desktop\\CT\\models\\run_' + str(run) + '\\'
    os.mkdir(model_path)
    
    folds = pd.read_csv('C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\folds_zeros.csv')
    
    train_params = {'batch_size' : 4,
                    'size_input' : 384,
                    'n_epochs': 1000,
                    'learning_rate': 5e-3,
                    'patience' : 5,
                    'delta' : 0,
                    'input_channels' : 1,
                    'output_channels' : 1}
    
    for split in range(5):
        
        model_path2 = os.path.join(model_path, f"split_{split}")
        os.mkdir(model_path2)
    
        train, valid, _ = get_sets(split, folds)
        
        train_dataset = TrainDataset(train, img_dir, per_dir, seg_dir, transform=TrainTransform(train_params['size_input']))
        valid_dataset = ValidDataset(valid, img_dir, per_dir, seg_dir, transform=ValidTransform(train_params['size_input']))
        
        world_size=2
        mp.spawn(main, args=(world_size, train_dataset, valid_dataset, train_params, model_path2, run, split), nprocs=world_size)

from can3d_multi_nores import CAN3D
    
if __name__ == "__main__":
    
    run = 85
    
    img_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\patient\\'
    per_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\pericardium\\'
    seg_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\calcium_multi\\'
    
    model_path = 'C:\\Users\\RuiSantos\\Desktop\\CT\\models\\run_' + str(run) + '\\'
    os.mkdir(model_path)
    
    folds = pd.read_csv('C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\folds_zeros.csv')
    
    train_params = {'batch_size' : 4,
                    'size_input' : 384,
                    'n_epochs': 1000,
                    'learning_rate': 5e-3,
                    'patience' : 5,
                    'delta' : 0,
                    'input_channels' : 1,
                    'output_channels' : 1}
    
    for split in range(5):
        
        model_path2 = os.path.join(model_path, f"split_{split}")
        os.mkdir(model_path2)
    
        train, valid, _ = get_sets(split, folds)
        
        train_dataset = TrainDataset(train, img_dir, per_dir, seg_dir, transform=TrainTransform(train_params['size_input']))
        valid_dataset = ValidDataset(valid, img_dir, per_dir, seg_dir, transform=ValidTransform(train_params['size_input']))
        
        world_size=2
        mp.spawn(main, args=(world_size, train_dataset, valid_dataset, train_params, model_path2, run, split), nprocs=world_size)
        
from can3d_multi_nobin import CAN3D
    
if __name__ == "__main__":
    
    run = 86
    
    img_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\patient\\'
    per_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\pericardium\\'
    seg_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\calcium_multi\\'
    
    model_path = 'C:\\Users\\RuiSantos\\Desktop\\CT\\models\\run_' + str(run) + '\\'
    os.mkdir(model_path)
    
    folds = pd.read_csv('C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\folds_zeros.csv')
    
    train_params = {'batch_size' : 4,
                    'size_input' : 384,
                    'n_epochs': 1000,
                    'learning_rate': 5e-3,
                    'patience' : 5,
                    'delta' : 0,
                    'input_channels' : 1,
                    'output_channels' : 1}
    
    for split in range(5):
        
        model_path2 = os.path.join(model_path, f"split_{split}")
        os.mkdir(model_path2)
    
        train, valid, _ = get_sets(split, folds)
        
        train_dataset = TrainDataset(train, img_dir, per_dir, seg_dir, transform=TrainTransform(train_params['size_input']))
        valid_dataset = ValidDataset(valid, img_dir, per_dir, seg_dir, transform=ValidTransform(train_params['size_input']))
        
        world_size=2
        mp.spawn(main, args=(world_size, train_dataset, valid_dataset, train_params, model_path2, run, split), nprocs=world_size)

from can3d_multi_noinit import CAN3D
    
if __name__ == "__main__":
    
    run = 87
    
    img_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\patient\\'
    per_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\pericardium\\'
    seg_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\calcium_multi\\'
    
    model_path = 'C:\\Users\\RuiSantos\\Desktop\\CT\\models\\run_' + str(run) + '\\'
    os.mkdir(model_path)
    
    folds = pd.read_csv('C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\folds_zeros.csv')
    
    train_params = {'batch_size' : 4,
                    'size_input' : 384,
                    'n_epochs': 1000,
                    'learning_rate': 5e-3,
                    'patience' : 5,
                    'delta' : 0,
                    'input_channels' : 1,
                    'output_channels' : 1}
    
    for split in range(5):
        
        model_path2 = os.path.join(model_path, f"split_{split}")
        os.mkdir(model_path2)
    
        train, valid, _ = get_sets(split, folds)
        
        train_dataset = TrainDataset(train, img_dir, per_dir, seg_dir, transform=TrainTransform(train_params['size_input']))
        valid_dataset = ValidDataset(valid, img_dir, per_dir, seg_dir, transform=ValidTransform(train_params['size_input']))
        
        world_size=2
        mp.spawn(main, args=(world_size, train_dataset, valid_dataset, train_params, model_path2, run, split), nprocs=world_size)

from can3d_multi_noskip_nodil import CAN3D
    
if __name__ == "__main__":
    
    run = 88
    
    img_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\patient\\'
    per_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\pericardium\\'
    seg_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\calcium_multi\\'
    
    model_path = 'C:\\Users\\RuiSantos\\Desktop\\CT\\models\\run_' + str(run) + '\\'
    os.mkdir(model_path)
    
    folds = pd.read_csv('C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\folds_zeros.csv')
    
    train_params = {'batch_size' : 4,
                    'size_input' : 384,
                    'n_epochs': 1000,
                    'learning_rate': 5e-3,
                    'patience' : 5,
                    'delta' : 0,
                    'input_channels' : 1,
                    'output_channels' : 1}
    
    for split in range(5):
        
        model_path2 = os.path.join(model_path, f"split_{split}")
        os.mkdir(model_path2)
    
        train, valid, _ = get_sets(split, folds)
        
        train_dataset = TrainDataset(train, img_dir, per_dir, seg_dir, transform=TrainTransform(train_params['size_input']))
        valid_dataset = ValidDataset(valid, img_dir, per_dir, seg_dir, transform=ValidTransform(train_params['size_input']))
        
        world_size=2
        mp.spawn(main, args=(world_size, train_dataset, valid_dataset, train_params, model_path2, run, split), nprocs=world_size)

from can3d_multi_noskip_nores import CAN3D
    
if __name__ == "__main__":
    
    run = 89
    
    img_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\patient\\'
    per_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\pericardium\\'
    seg_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\calcium_multi\\'
    
    model_path = 'C:\\Users\\RuiSantos\\Desktop\\CT\\models\\run_' + str(run) + '\\'
    os.mkdir(model_path)
    
    folds = pd.read_csv('C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\folds_zeros.csv')
    
    train_params = {'batch_size' : 4,
                    'size_input' : 384,
                    'n_epochs': 1000,
                    'learning_rate': 5e-3,
                    'patience' : 5,
                    'delta' : 0,
                    'input_channels' : 1,
                    'output_channels' : 1}
    
    for split in range(5):
        
        model_path2 = os.path.join(model_path, f"split_{split}")
        os.mkdir(model_path2)
    
        train, valid, _ = get_sets(split, folds)
        
        train_dataset = TrainDataset(train, img_dir, per_dir, seg_dir, transform=TrainTransform(train_params['size_input']))
        valid_dataset = ValidDataset(valid, img_dir, per_dir, seg_dir, transform=ValidTransform(train_params['size_input']))
        
        world_size=2
        mp.spawn(main, args=(world_size, train_dataset, valid_dataset, train_params, model_path2, run, split), nprocs=world_size)

from can3d_multi_noskip_nobin import CAN3D
    
if __name__ == "__main__":
    
    run = 90
    
    img_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\patient\\'
    per_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\pericardium\\'
    seg_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\calcium_multi\\'
    
    model_path = 'C:\\Users\\RuiSantos\\Desktop\\CT\\models\\run_' + str(run) + '\\'
    os.mkdir(model_path)
    
    folds = pd.read_csv('C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\folds_zeros.csv')
    
    train_params = {'batch_size' : 4,
                    'size_input' : 384,
                    'n_epochs': 1000,
                    'learning_rate': 5e-3,
                    'patience' : 5,
                    'delta' : 0,
                    'input_channels' : 1,
                    'output_channels' : 1}
    
    for split in range(5):
        
        model_path2 = os.path.join(model_path, f"split_{split}")
        os.mkdir(model_path2)
    
        train, valid, _ = get_sets(split, folds)
        
        train_dataset = TrainDataset(train, img_dir, per_dir, seg_dir, transform=TrainTransform(train_params['size_input']))
        valid_dataset = ValidDataset(valid, img_dir, per_dir, seg_dir, transform=ValidTransform(train_params['size_input']))
        
        world_size=2
        mp.spawn(main, args=(world_size, train_dataset, valid_dataset, train_params, model_path2, run, split), nprocs=world_size)
        
from can3d_multi_noskip_noinit import CAN3D
    
if __name__ == "__main__":
    
    run = 91
    
    img_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\patient\\'
    per_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\pericardium\\'
    seg_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\calcium_multi\\'
    
    model_path = 'C:\\Users\\RuiSantos\\Desktop\\CT\\models\\run_' + str(run) + '\\'
    os.mkdir(model_path)
    
    folds = pd.read_csv('C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\folds_zeros.csv')
    
    train_params = {'batch_size' : 4,
                    'size_input' : 384,
                    'n_epochs': 1000,
                    'learning_rate': 5e-3,
                    'patience' : 5,
                    'delta' : 0,
                    'input_channels' : 1,
                    'output_channels' : 1}
    
    for split in range(5):
        
        model_path2 = os.path.join(model_path, f"split_{split}")
        os.mkdir(model_path2)
    
        train, valid, _ = get_sets(split, folds)
        
        train_dataset = TrainDataset(train, img_dir, per_dir, seg_dir, transform=TrainTransform(train_params['size_input']))
        valid_dataset = ValidDataset(valid, img_dir, per_dir, seg_dir, transform=ValidTransform(train_params['size_input']))
        
        world_size=2
        mp.spawn(main, args=(world_size, train_dataset, valid_dataset, train_params, model_path2, run, split), nprocs=world_size)

from unet_2d_3d import UNet3D

def main(rank, world_size, train_dataset, valid_dataset, params, path, run, split):
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, drop_last=True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_dl = DataLoader(train_dataset, params['batch_size'], sampler=train_sampler, drop_last=True) 
    valid_dl = DataLoader(valid_dataset, params['batch_size'], sampler=valid_sampler, drop_last=True)
                
    model = UNet3D().to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    train_model(params, model, train_dl, valid_dl, path, rank, world_size, run, split)

    dist.destroy_process_group()
    
if __name__ == "__main__":
    
    run = 92
    
    img_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\patient\\'
    per_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\pericardium\\'
    seg_dir = 'C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\calcium_multi\\'
    
    model_path = 'C:\\Users\\RuiSantos\\Desktop\\CT\\models\\run_' + str(run) + '\\'
    os.mkdir(model_path)
    
    folds = pd.read_csv('C:\\Users\\RuiSantos\\Desktop\\CT\\datasets\\COCA\\folds_zeros.csv')
    
    train_params = {'batch_size' : 1,
                    'size_input' : 384,
                    'n_epochs': 1000,
                    'learning_rate': 5e-4,
                    'patience' : 5,
                    'delta' : 0,
                    'input_channels' : 1,
                    'output_channels' : 1}
    
    for split in range(5):
        
        model_path2 = os.path.join(model_path, f"split_{split}")
        os.mkdir(model_path2)
    
        train, valid, _ = get_sets(split, folds)
        
        train_dataset = TrainDataset(train, img_dir, per_dir, seg_dir, transform=TrainTransform(train_params['size_input']))
        valid_dataset = ValidDataset(valid, img_dir, per_dir, seg_dir, transform=ValidTransform(train_params['size_input']))
        
        world_size=2
        mp.spawn(main, args=(world_size, train_dataset, valid_dataset, train_params, model_path2, run, split), nprocs=world_size)