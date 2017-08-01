#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 18:40:17 2017

@author: wuzhenglin
"""

import numpy as np 
import pandas as pd

def read_data_contrastAndimage():
    
    path_csv = '/Users/wuzhenglin/Python_nice/SAL_LUNG/siim-medical-image-analysis-tutorial/overview.csv'
    overview_df = pd.read_csv(path_csv)
    overview_df.columns = ['idx']+list(overview_df.columns[1:])
    overview_df['Contrast'] = overview_df['Contrast'].map(lambda x: 1 if x else 0)
    
    path_np = '/Users/wuzhenglin/Python_nice/SAL_LUNG/siim-medical-image-analysis-tutorial/full_archive.npz'
    im_data = np.load(path_np)
    full_image_dict = dict(zip(im_data['idx'], im_data['image']))
    
    for x in full_image_dict.keys():
        full_image_dict[x] = (full_image_dict[x] - full_image_dict[x].min()) \
        / (full_image_dict[x].max() - full_image_dict[x].min()) * 255
        
        full_image_dict[x] = full_image_dict[x][::2,::2]

     
    labels = dict(zip(overview_df['idx'],overview_df['Contrast']))
    train_data = np.asarray([full_image_dict[x].flatten() for x in list(full_image_dict.keys())[:400] if len(full_image_dict[x].flatten()) == 256*256])
    train_labels = np.asarray([labels[x] for x in list(full_image_dict.keys())[:400] if len(full_image_dict[x].flatten()) == 256*256])
    
    tmp = np.zeros((train_labels.shape[0],2))
    for i,x in enumerate(tmp):
        if train_labels[i] == 0:
            tmp[i][0] = 1
        else:
            tmp[i][1] = 1
    train_labels = tmp
    
    test_data = np.asarray([full_image_dict[x].flatten() for x in list(full_image_dict.keys())[-75:] if len(full_image_dict[x].flatten()) == 256*256])
    test_labels = np.asarray([labels[x] for x in list(full_image_dict.keys())[-75:] if len(full_image_dict[x].flatten()) == 256*256])
   
    tmp = np.zeros((test_labels.shape[0],2))
    for i,x in enumerate(tmp):
        if test_labels[i] == 0:
            tmp[i][0] = 1
        else:
            tmp[i][1] = 1
    test_labels = tmp
    
    return train_data, train_labels, test_data, test_labels
    
    
    
    
    
    
    
    

if __name__ == '__main__':
    
    read_data_contrastAndimage()
    