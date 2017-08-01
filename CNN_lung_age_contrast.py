#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 01:38:36 2017

@author: wuzhenglin
"""
#==============================================================================
# same as CNN_lung_image_mask.py, only different on the output and train parts
#==============================================================================

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from lung_age_contrast import read_data_contrastAndimage


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
    
    X_train, X_test, y_train, y_test = read_data_contrastAndimage()
    
    epochs = 10
    batch_size = 20
    learning_rate = 0.01
    hidden = 100
    
    image_size = 256
    label_size = 2
   
    x = tf.placeholder(tf.float32, shape = [None, image_size * image_size])
    y = tf.placeholder(tf.float32, shape = [None, label_size])
  
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
    W_fc1 = weight_variable([32 * 32 * 64, hidden])
    b_fc1 = bias_variable([hidden])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 32 * 32 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    #Output_Softmax
    
    W_fc2 = weight_variable([hidden, label_size])
    b_fc2 = bias_variable([label_size])
    
    y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

    #Train
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_conv, labels=y))
    #cross_entropy = -tf.reduce_sum(y * tf.log(y_conv))
    optimize = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #sess
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        
        
        count = 1
        itr = X_train.shape[0] // batch_size        
        
        for num_1 in range(epochs):
            i = 0
            for num_2 in range(itr):
                X_train_batch = X_train[i : batch_size * (num_2 + 1)]          
                y_train_batch = y_train[i : batch_size * (num_2 + 1)]
              
                i = i + batch_size
                
                
                feed_dict = {x: X_train_batch, y: y_train_batch}
                
                
                sess.run(optimize, feed_dict = feed_dict)
                loss, acc = sess.run([cross_entropy, accuracy], feed_dict = feed_dict)
            
                print "step: %d  loss: %.9f  accuracy: %.3f" % (count, loss, acc)
                
                count = count + 1
                
        
        print 'Finish Train.'
        
#        X_test_batch = X_test[0]
#        
#        y_test_batch = y_test[0]
#        
#        
#        feed_dict_3 = {x: X_test_batch, y: y_test_batch, keep_prob: 1.0}
#        sess.run(optimize, feed_dict = feed_dict_3)
#        
#        loss_, acc_ = sess.run([cross_entropy, accuracy], feed_dict = feed_dict_3)
#            
#        print "TEST:  loss: %.9f  accuracy: %.3f" % (loss_, acc_)
#        print y_conv.shape
#        print y_conv
#        
#        pred = sess.run(y_conv, feed_dict = feed_dict_3)
#        print pred
#        
#        fig, (ax1, ax2, ax3) = plt.subplots(1,3)    
#        ax1.imshow(X_test_batch[0])
#        ax2.imshow(y_test_batch[0])
#        ax3.imshow(pred)
#        

if __name__ == '__main__':
    
    cnn_model()