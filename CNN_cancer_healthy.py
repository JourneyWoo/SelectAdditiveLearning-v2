#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:33:54 2017

@author: wuzhenglin
"""

import numpy as np
import tensorflow as tf
import dicom as dm
import matplotlib.pyplot as plt
from lung_healthy_cancer import make


def read_and_decode():
    
    filename = '/Users/wuzhenglin/Python_nice/SAL_LUNG/train.tfrecords'
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    
    
    print '*******************************************'
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    print img
    print img.shape
    img = tf.reshape(img, [512, 512])

    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.int32)
    print '*******************************************'
    return img, label

def weight_variable(shape):
    
    initial = tf.truncated_normal(shape, stddev = 0.1, dtype = tf.float32)
    return tf.Variable(initial)


def bias_variable(shape):
    
    initial = tf.constant(0.1, shape = shape, dtype = tf.float32)
    return tf.Variable(initial)


def conv2d(x, W):
    
    #(input, filter, strides, padding)
    #[batch, height, width, in_channels]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    
    #(value, ksize, strides, padding)
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  

def cnn_model():
    
    epochs = 1
    batch_size = 30
    learning_rate = 0.01
    hidden = 100
    cap_c = 4682
    cap_h = 5097
    
    image_size = 512
    label_size = 2
    
#    X_train, y_train = read_and_decode()
    all_images,_ , all_cancer_images, all_healthy_images= make()
    
    Train_c = np.stack([dm.read_file(img_path_1).pixel_array.flatten() for img_path_1 in all_cancer_images], 0)
    Train_h = np.stack([dm.read_file(img_path_2).pixel_array.flatten() for img_path_2 in all_healthy_images], 0)
    
    Label_cc = []
    labelc = 1
    
    for ii in range(cap_c):
        Label_cc.append(labelc)
        
    Label_c = np.asarray(Label_cc, dtype = np.int32)
        
    Label_hh = []
    labelh = 0
    
    for iii in range(cap_h):
        Label_hh.append(labelh)
    Label_h = np.asarray(Label_hh, dtype = np.int32)
    
    print Label_c.shape
    print Label_h.shape
    

    
   
    x = tf.placeholder(tf.float32, shape = [None, image_size * image_size])
    y = tf.placeholder(tf.int32, shape = [None,])
  
    X_train_ = tf.reshape(x, [-1, image_size, image_size, 1])

    #First layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
      
    h_conv1 = tf.nn.relu(conv2d(X_train_, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    #Second layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    #Full connect layer
    W_fc1 = weight_variable([128 * 128 * 64, hidden])
    b_fc1 = bias_variable([hidden])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 128 * 128 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    #Output_Softmax
    
    W_fc2 = weight_variable([hidden, label_size])
    b_fc2 = bias_variable([label_size])
    
    y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

    #Train
    loss = tf.losses.sparse_softmax_cross_entropy(labels = y, logits = y_conv)
    #cross_entropy = -tf.reduce_sum(y * tf.log(y_conv))
    optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
#    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    correct_prediction = tf.equal(tf.cast(tf.argmax(y_conv, 1), tf.int32), y) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
#    img_batch, label_batch = tf.train.shuffle_batch([X_train, y_train],
#                                                batch_size = batch_size, capacity = cap,
#                                                min_after_dequeue = 2000)
    
    #Since tfrecord always error, in this train I will use the separate method
    
    
        
    
    
    print Train_c[0].shape
    print 'Begin training'
    #sess
    init = tf.global_variables_initializer()
   
    with tf.Session() as sess:
        sess.run(init)
        
#        coord = tf.train.Coordinator()
#        threads = tf.train.start_queue_runners(coord = coord)
        
        count_c = cap_c // batch_size
        count_h = cap_h // batch_size
        
        step = 1
        pointer_c = 0
        pointer_h = 0
        c = 0
        h = 0
        
        for ep in range(epochs):
            
            
#            train, lab= sess.run([img_batch, label_batch])
            for p in range(count_c + count_h):
                
                if p < count_c:
                
                    feed_dict = {x: Train_c[pointer_c: batch_size * (c + 1)], y: Label_c[pointer_c: batch_size * (c + 1)]}
                    pointer_c = pointer_c + batch_size
                    c = c + 1
                    
                else:
                    
                    feed_dict = {x: Train_h[pointer_h: batch_size * (h + 1)], y: Label_h[pointer_h: batch_size * (h + 1)]}
                    pointer_h = pointer_h + batch_size
                    h = h + 1
                    
                sess.run(optimize, feed_dict = feed_dict)
                
                los, acc = sess.run([loss, accuracy], feed_dict = feed_dict)
            
                print "step: %d  loss: %.9f  accuracy: %.3f" % (step, los, acc)
            
                step = step + 1
                
            
        
                
        
        print 'Finish Train.'
               

if __name__ == '__main__':
    
    cnn_model()
