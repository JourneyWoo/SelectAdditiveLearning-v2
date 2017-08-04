#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 00:56:57 2017

@author: wuzhenglin
"""

import numpy as np 
import pandas as pd
import os
from glob2 import glob
import tensorflow as tf
import dicom as dm

def folder_travel(s, folder_path):
    
    files = os.listdir(folder_path)
    
    for each in files:
        
        if (each[0] == '.'):
            pass
        
        else:
        
            flag = os.path.isdir(os.path.join(folder_path, each))
        
            if flag:
                path = folder_path + '/' + each
                s = folder_travel(s, path)
                
            else:
                f = folder_path + '/' + each
                iter_f = iter(f)
                str = ''
                for line in iter_f:
                    str = str + line
                s.append(str)
            
    
    return s
        
def make():
    
    cancer_list = []
    cancer_image_path = '/Users/wuzhenglin/Python_nice/SAL_LUNG/lung_cancer_CT/DOI'
    all_cancer_images = folder_travel(cancer_list, cancer_image_path)
    len_all_cancer_images = len(all_cancer_images)
    print 'OK, Find ', len_all_cancer_images, ' cancer images'
    
    healthy_list = []
    healthy_image_path = '/Users/wuzhenglin/Python_nice/SAL_LUNG/healthy_lung_CT'
    all_healthy_images = folder_travel(healthy_list, healthy_image_path)
    len_all_healthy_images = len(all_healthy_images)
    print 'OK, Find ', len_all_healthy_images, ' healthy images'
    
    all_images = all_cancer_images + all_healthy_images
    len_all_images = len(all_images)
    print 'Merge finish, there are ', len_all_images, ' cancer and healthy images'
    
    
    cwd = os.getcwd()
    print '************ CWD:', cwd, ' ************'
    return all_images, len_all_cancer_images, all_cancer_images, all_healthy_images
    
def make_dataset():
    
    all_images, len_all_cancer_images,_,_ = make()
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for idx, img_path in enumerate(all_images):
        img = dm.read_file(img_path)
        pix = img.pixel_array
#        pixel_bytes = img.PixelData 
        pix_byte = pix.tostring()
        img_raw = pix_byte
        if idx < len_all_cancer_images:
            example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list = tf.train.Int64List(value = [1])),
            'img_raw': tf.train.Feature(bytes_list = tf.train.BytesList(value = [img_raw]))}))
            writer.write(example.SerializeToString())
        
        else:
            example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list = tf.train.Int64List(value=[0])),
            'img_raw': tf.train.Feature(bytes_list = tf.train.BytesList(value = [img_raw]))}))
            writer.write(example.SerializeToString())
    
    writer.close()
    print 'Finish?!'
    


if __name__ == '__main__':
    
    make()
    