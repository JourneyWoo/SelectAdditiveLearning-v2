#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:04:40 2017

@author: wuzhenglin
"""


import numpy as np

import os
from glob2 import glob
from skimage.io import imread

from sklearn.model_selection import train_test_split


def extract_image(image_path_1, image_path_2, image_name, image_type):
   
    print 'Extracting the images...'
    BASE_IMAGE_PATH = os.path.join(image_path_1, image_path_2)
    all_images = glob(os.path.join(BASE_IMAGE_PATH, image_name, image_type))
    return all_images

def read_data_maskAndimage():
    
    image_path_1 = '/Users/wuzhenglin/Python_nice/SAL_LUNG'
    image_path_2 = 'finding-lungs-in-ct-data'
    image_name_img = '2d_images'
    image_name_msk = '2d_masks'
    image_type = '*.tif'
    
    images = extract_image(image_path_1, image_path_2, image_name_img, image_type)
    print len(images), 'matching image files found'
    masks = extract_image(image_path_1, image_path_2, image_name_msk, image_type)
    print len(masks), 'matching mask files found'
    
    print '******Extracting Finish******\n'
    
    jimread = lambda x: np.expand_dims(imread(x)[::4, ::4],0)
    test_image = jimread(images[0])
#    test_mask = jimread(masks[0])
#    fig, (ax1 ,ax2) = plt.subplots(1, 2)
#    ax1.imshow(test_image[0])
#    ax2.imshow(test_mask[0])
    
    print 'Reading the images...'
    print 'Total samples are', len(images)
    print 'Image resolution is', test_image[0].shape  
    
    imageTrain = np.stack([jimread(img) for img in images], 0)
    maskLabel = np.stack([jimread(msk) for msk in masks], 0) / 255.0
    X_train, X_test, y_train, y_test = train_test_split(imageTrain, maskLabel, test_size=0.1)
    
    print 'Training input is', X_train.shape  
    print 'Training output is {}, min is {}, max is {}'.format(y_train.shape, y_train.min(), y_train.max())
    print 'Testing set is', X_test.shape
    
    print '******Reading Finish*******\n'
    
    return X_train, X_test, y_train, y_test

               
                


if __name__ == '__main__':
    
    read_data_maskAndimage()
    