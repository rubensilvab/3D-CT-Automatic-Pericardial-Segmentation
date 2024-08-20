# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:00:43 2024

@author: RubenSilva
"""

"Funçoes pos processamento"


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import shutil
import openpyxl
import cv2
import pydicom
import cc3d
import torch

def fill_3d(labels_out):
    mask_convex=np.zeros(labels_out.shape)
    for i in range(labels_out.shape[0]):
        if( i==0 or i==labels_out.shape[0]-1):
            mask_convex[i,:,:]=labels_out[i,:,:]
        else:
            mask_convex[i,:,:]=np.logical_or(labels_out[i,:,:],np.logical_and(labels_out[i-1,:,:], labels_out[i+1,:,:]))
    return mask_convex.astype(np.uint8)

def connected_components(pred_test):
     # Get a labeling of the k largest objects in the image.
     # The output will be relabeled from 1 to N.
     labels_out, N = cc3d.largest_k(
       pred_test, k=1, 
       connectivity=6, delta=0,
       return_N=True,
     )
    
     labels_out=labels_out.astype(np.uint8)
     
     return labels_out
 
import cv2
import skimage.morphology, skimage.data

def fill_holes(labels_out):
    mask_fill=labels_out.copy()
    filled_img = np.zeros_like(mask_fill)
    for i in range(labels_out.shape[0]):
      # Find contours
        contours, _ = cv2.findContours(mask_fill[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on black image
        contour_img = np.zeros_like(mask_fill[i])
        cv2.drawContours(contour_img, contours, -1, 1, 1)
        
        # Fill enclosed regions
        
        for contour in contours:
            cv2.fillPoly(filled_img[i], [contour], 1)
          
    return filled_img.astype(np.uint8)


from scipy.spatial import ConvexHull

from PIL import Image, ImageDraw

def convex_hull_image(data):
    w,l=data.shape[0],data.shape[1]
    region = np.argwhere(data)
    try:   
        hull = ConvexHull(region)
        verts = [(region[v,0], region[v,1]) for v in hull.vertices]
        img = Image.new('L', data.shape, 0)
        ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
        mask = np.array(img)
    except:    
        mask=np.zeros((w,l))
    return mask.T

def convex_mask(labels_out):
    mask_convex=np.zeros(labels_out.shape)
    for i in range(labels_out.shape[0]):
            mask_convex[i,:,:]=convex_hull_image(labels_out[i,:,:])
    return mask_convex.astype(np.uint8)


def pos_process(pred,device):
    pred=pred.cpu().numpy()
    pred=np.squeeze(pred)
    connected=connected_components(pred)   
    
    #convex_predict=fill_3d(connected)
    fill_2d_predict=fill_3d(connected)
    
    #convex_predict=convex_mask(convex_predict)
    fill_2d_predict=fill_holes(fill_2d_predict)
    
    fill_2d_predict = torch.from_numpy(fill_2d_predict).to(device)
    fill_2d_predict=torch.unsqueeze(fill_2d_predict, 0)
    fill_2d_predict=torch.unsqueeze(fill_2d_predict, 0)
   
    return fill_2d_predict
