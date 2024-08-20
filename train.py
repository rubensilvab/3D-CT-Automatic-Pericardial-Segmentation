# -*- coding: utf-8 -*-
"""
Created on Wed May 15 18:24:17 2024

@author: RubenSilva
"""

# set the matplotlib backend so figures can be saved in the background
import matplotlib
#matplotlib.use("Agg")
# import the necessary packages
from LeNet import LeNet
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
import cv2
from PIL import Image

#Import Custom DataSet

import os
from torch.utils.data import Dataset
from torchvision.io import read_image
from New_CustomDataset import *


if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))  



import pandas as pd    


def dice_coefficient(output, target, class_index):
    
    smooth = 1e-5
    # Select the probabilities for the given class index
    output_class = output[class_index]
    target_class = (target == class_index).float()
    #print(target_class.size(),output_class.size())
    #output_class=target_class
    output_flat = output_class.view(-1)
    target_flat = target_class.view(-1)
    #print(target_flat.size(),output_flat.size())
    intersection = torch.sum(output_flat * target_flat)
    union = torch.sum(output_flat) + torch.sum(target_flat)
    dice = (2. * intersection + smooth) / (union + smooth)
    
    #print(dice)
    return dice

# def dice_loss(output, target, num_classes,batch_size,device):
#     loss = torch.zeros(batch_size).to(device)
    
#     for b in range(batch_size):        
#         for class_index in range(num_classes):
#             dice= 1 - dice_coefficient(output[b], target[b], class_index)
#             loss[b] += 0.5 * dice
            
#     return loss.mean()

def dice_loss(output,target, batch_size,my_device=my_device):

    loss = torch.zeros(batch_size).to(my_device)

    for i in range(batch_size):
        #dice_foreground = torch.sum(target[i] * output[i]) / (torch.sum(target[i]**2) + torch.sum(output[i]**2) + 1e-7)
        smooth = 1e-8
        output_flat = output[i].view(-1)
        target_flat = target[i].view(-1)
        intersection = torch.sum(target_flat * output_flat)
        union = torch.sum(output_flat) + torch.sum(target_flat)
        dice_foreground = (2. * intersection) / (union + smooth)
        
        #dice_background = torch.sum((1-target[i]) * (1-output[i])) / (torch.sum((1-target[i])**2) + torch.sum((1-output[i])**2) + 1e-7)
        #print(dice_foreground)
        loss[i] = 1 - dice_foreground #- dice_background

    return loss.mean()

def dice_loss2d(output,target, batch_size,my_device=my_device):

    loss = torch.zeros(batch_size).to(my_device)

    for i in range(batch_size):
        #dice_foreground = torch.sum(target[i] * output[i]) / (torch.sum(target[i]**2) + torch.sum(output[i]**2) + 1e-7)
        loss_t=0
        for s in range(output.shape[2]):

            smooth = 1e-8
            output_flat = output[i][:,s].view(-1)
            target_flat = target[i][:,s].view(-1)
            intersection = torch.sum(target_flat * output_flat)
            union = torch.sum(output_flat) + torch.sum(target_flat)
            dice_foreground = (2. * intersection) / (union + smooth)
            loss_s= 1 - dice_foreground
            loss_t+=loss_s
            
        loss_t=loss_t/output.shape[2]    
        #dice_background = torch.sum((1-target[i]) * (1-output[i])) / (torch.sum((1-target[i])**2) + torch.sum((1-output[i])**2) + 1e-7)
        #print(dice_foreground)
        loss[i] = loss_t #- dice_background

    return loss.mean()

def CheckTime():
    # get the current time in seconds since the epoch
    seconds = time.time()
    # convert the time in seconds since the epoch to a readable format
    local_times = time.ctime(seconds)
    local_time = '_'.join(local_times.split())
    local_time = '_'.join(local_time.split(':'))
    
    return local_time

def SaveTrainingState(epoch,hyperparameters,Info,opt):
    # Save the training state
    checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': opt.state_dict(),
    'hyperparameters': hyperparameters,
    'Info:': Info,
    'epoch': epoch 
    }
    return checkpoint

import gc

def train_model(model,hyperparameters,Info,Train_dl,Val_dl,path_to_save):

    # calculate steps per epoch for training and validation set
    #trainSteps = len(Train_dl.dataset) // hyperparameters['batch_size']
    #valSteps = len(Val_dl.dataset) // hyperparameters['batch_size']
    
    # initialize our optimizer and loss function
    opt = Adam(model.parameters(), lr=hyperparameters['learning_rate'])
    
    local_time=CheckTime()
    path_model=os.path.join(path_to_save,local_time)
    
    if not os.path.exists(path_model):                         
            # Create a new directory because it does not exist 
            os.makedirs(path_model)     
    os.chdir(path_model) 
        
    best_model_path=os.path.join(path_model,local_time+'.pth')
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('C:/Users/RubenSilva/Desktop/pytorch_test/log/'+local_time) 
    # measure how long training is going to take
    print("[INFO] training the network...")
    
    EPOCHS = hyperparameters['epochs']
    best_val_loss=float('+inf')  # Initialize the best validation score
    count=-1
    # loop over our epochs
    for e in range(0, EPOCHS):
        # set the model in training mode
        model.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0       
        tenstepsTrainLoss=0
        
        # loop over the training set
        for i, data in enumerate(Train_dl,0):
            #the input to the device
            img, mask= data
            #Random ZCrop
            img, mask=RandomCrop(img, mask)
            img, mask = img.to(my_device), mask.to(my_device)
            
            # perform a forward pass
            pred = model(img)
            
            # calculate the training loss
            actual_batch=pred.shape[0]
            loss = dice_loss(pred,mask ,actual_batch,my_device)
            
            # zero out the gradients, perform the backpropagation step, and update the weights
            opt.zero_grad()
            loss.backward()        
            opt.step()
        
            # add the loss to the total training loss so far and calculate the number of correct predictions
            #print('Já correu opt')
            #startTimepred = time.time() 
            totalTrainLoss += loss.item()  # Assuming loss is a tensor
            #endtimepred= time.time() - startTimepred
            #print('Time to calculate totalTrainLoss += loss.item()', endtimepred)
            #print('Step:',steps,loss.item())
        
            tenstepsTrainLoss += loss.item() 
            if i % 10 == 9:    # every 9 mini-batches...
                # ...log the running loss
                writer.add_scalar('training loss', tenstepsTrainLoss / 10, e * len(Train_dl) + i)
                tenstepsTrainLoss=0
                #print('entrou 20 mini',i)    
            # switch off autograd for evaluation
            
            # Free up memory
            del img
            del mask
            del pred
            
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            tenstepsTrainLoss=0
            for i, data in enumerate(Val_dl,0):
     	        # send the input to the device
                img, mask=data
                #Random ZCrop
                img, mask=RandomCrop(img, mask)
                img, mask = img.to(my_device), mask.to(my_device)
     	        # make the predictions and calculate the validation loss
       	        pred = model(img)
                actual_batch=pred.shape[0]
                totalValLoss += dice_loss(pred,mask ,actual_batch,my_device)
                
                tenstepsTrainLoss += dice_loss(pred,mask ,actual_batch,my_device) 
                if i % 10 == 9:    # every 9 mini-batches...
                    writer.add_scalar('Validation loss', tenstepsTrainLoss / 10, e * len(Val_dl) + i) 
                    tenstepsTrainLoss=0
                
                # Free up memory
                del img
                del mask
                del pred

        #calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / len(Train_dl)
        avgValLoss = totalValLoss / len(Val_dl)
        
        # ...log the Average Training loss
        writer.add_scalar('Average Training loss', avgTrainLoss, e+1)
        # ...log the Average Validation Loss
        writer.add_scalar('Average Validation loss', avgValLoss, e+1) 
        
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
        print("Train loss: {:.6f}, Validation loss: {:.6f}".format(avgTrainLoss,avgValLoss))
        
       # Check if the current validation score is the best
               
        if avgValLoss < best_val_loss:
            best_val_loss = avgValLoss
            model_state=SaveTrainingState(e,hyperparameters,Info,opt)
            torch.save(model_state, best_model_path)  # Save the best model
            print(f'Best model saved with validation loss: {best_val_loss:.4f}')
            count=0
        else:    
            count += 1
            print('Average validation loss did not improve for ',count,' times.')
        
        
        if count == hyperparameters['patience']:
            
            writer.flush()
            writer.close()
            return None    
    
    # Empty the cache and collect garbage
    torch.cuda.empty_cache()
    gc.collect()      
           
    writer.flush()
    writer.close()
    
    return None    


"""Define train and val sets""" 

# Replace 'file_path.csv' with the path to your CSV file
file_path_abd = 'C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/Abdominal_5.csv'
file_path_cfat= 'C:/Users/RubenSilva/Desktop/CardiacCT_peri/CardiacFat/CFAT_5.csv'
file_path_osic='E:/RubenSilva/PericardiumSegmentation/Dataset/OSIC/OSIC_5.csv'

# Read the CSV file into a DataFrame
csv_file_abd = pd.read_csv(file_path_abd)
csv_file_cfat = pd.read_csv(file_path_cfat)
csv_file_osic = pd.read_csv(file_path_osic)

# Concatenate the DataFrames and reset the index
csv_file = pd.concat([csv_file_abd, csv_file_cfat, csv_file_osic], ignore_index=True)
csv_file.loc[:,'Patient']=csv_file['Patient'].astype(str)
  
"""Choose fold for train, val and test"""
folds_train=[1,2,0]
folds_val=[3]
folds_test=[4]

from torchvision import transforms

# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
# ])

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomApply([ transforms.RandomRotation(5)], p=0.25),
    transforms.RandomApply([transforms.RandomResizedCrop(256, scale=(0.7,0.9))], p=0.25),
    #transforms.GaussianBlur(kernel_size=5, sigma=(1.0,1.2))
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(1.0,1.2))], p=0.25)
    #AddGaussianNoise(mean=0.0, std=0.05)
])


TrainSetcsv=csv_file.loc[csv_file['Fold'].isin(folds_train)]
ValSetcsv=csv_file.loc[csv_file['Fold'].isin(folds_val)]    

Info= {
    'Name model': 'CAN3D No Skip',
    'Dataset:': 'CHVNGE(Abdominal), CFat, OSIC',
    'Loss': 'Dice Loss' ,
    'Augmentations': str(transform),
    'Padding': 'No',
    'ZCrop': 'Yes',
    'img_size': 256,
    'folds_train': str(folds_train),
    'folds_val': str(folds_val),
    'folds_test': str(folds_test),
    'Patients_train (total):': str(len(TrainSetcsv)),
    'Patients_val (total):':str(len(ValSetcsv))
       
}

# Hyperparameters and other info
hyperparameters = {
    'learning_rate': 0.001,
    'epochs': 1000,
    'batch_size': 2,
    'patience':8
}

   
TrainDatset = PatientCustomDatasetCSVdcm(TrainSetcsv,transform=transform)
ValDataset = PatientCustomDatasetCSVdcm(ValSetcsv,transform=transform)

"""Load SlicePind to locate the patients with the same CT Slices"""
ind_train,N_slices_train,SliPind_train=ExtractSliceInd(TrainSetcsv)
ind_val,N_slices_val,SliPind_val=ExtractSliceInd(ValSetcsv)
    
train_sampler=SubsetSampler(ind_train,N_slices_train,SliPind_train,batch_size=hyperparameters['batch_size'])
val_sampler=SubsetSampler(ind_val,N_slices_val,SliPind_val,batch_size=hyperparameters['batch_size'])

"""Load DataLoader"""
Train_dl = DataLoader(TrainDatset, batch_sampler=train_sampler)
Val_dl=  DataLoader(ValDataset, batch_sampler=val_sampler)


from torch.utils.tensorboard import SummaryWriter

"""Import model"""
from can3d_multi_noskip import CAN3D 

print("[INFO] initializing the CAN3D model...")
model=CAN3D().to(my_device)

path_to_save='C:/Users/RubenSilva/Desktop/pytorch_test/models'

train_model(model,hyperparameters,Info,Train_dl,Val_dl,path_to_save)

# #Display image and label.
# train_features, train_labels = next(iter(Train_dl))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")

# for i in range(64):
#     img = train_features[0].squeeze()[i]
#     label = train_labels[0].squeeze()[i]
#     plt.imshow(img, cmap="gray")
#     plt.colorbar()
#     plt.imshow(label,cmap="gray",alpha=0.5)

#     plt.show()
#     print(np.unique(img))


    
#     # finish measuring how long training took


# # convert the time in seconds since the epoch to a readable format
# local_time_end = CheckTime()


# f= open(os.path.join(Path_to_save_model,local_time,local_time+".txt"),"w+")
  
# f.write("Date start: "+(local_time)+'\n')
# f.write("Date end: "+(local_time_end)+'\n')
# f.write("Model:"+Info['Name model']+'\n')
# f.write("Dataset: "+ Info['Dataset']+'\r\n')
# f.write("Cross-Validation ? "+str(cross_validation)+'\r\n')
# f.write("Folds used to train: "+str(folds_train)+'\n')
# f.write("Images used to train: "+ str(len(Train_dl.dataset))+'\r\n')
# f.write("Folds used to validation: "+str(folds_val)+'\n')
# f.write("Images used to validation: "+ str(len(Val_dl.dataset))+'\r\n')
# f.write("Folds used to test: "+str(folds_test)+'\n')
# f.write("Images used to test: Cardiac Fat: "+ str(len(test_cfat))+" + OSIC: "+str(len(test_osic))+'\r\n')
# f.write("Size: %d\n" % (Info['img_size']))
# f.write("Loss: "+Info['Loss']+'\n')
# f.write("Learning rate: "+str(Info['lr'])+'\n')
# f.write("Batch size:"+ str(Info['batch_size']) +'\n')
# f.write("Data augmentation: "+Info['Augmentations'])
# f.close() 




  
# # we can now evaluate the network on the test set
# print("[INFO] evaluating network...")
# # turn off autograd for testing evaluation
# with torch.no_grad():
# 	# set the model in evaluation mode
# 	model.eval()
# 	
# 	# initialize a list to store our predictions
# 	preds = []
# 	# loop over the test set
# 	for (img, mask) in testDataLoader:
# 		# send the input to the device
# 		img = x.to(device)
# 		# make the predictions and add them to the list
# 		pred = model(img)
# 		preds.extend(pred.argmax(axis=1).cpu().numpy())
# # generate a classification report
# print(classification_report(testData.targets.cpu().numpy(),
# 	np.array(preds), target_names=testData.classes))

# plot the training loss and accuracy


