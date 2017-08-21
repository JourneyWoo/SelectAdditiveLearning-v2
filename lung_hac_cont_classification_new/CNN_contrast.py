#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 03:54:23 2017

@author: wuzhenglin
"""

import numpy as np
import tensorflow as tf
import dicom as dm
import matplotlib.pyplot as plt


def change(arr):
    
    
    inti = arr[0]
    
    if inti == 0:
        inti = np.array([1, 0])
    else:
        inti = np.array([0, 1])
    
    inti = inti[np.newaxis, :] 
    
    for i in range(1, arr.shape[0]):
        
        tem = arr[i]
    
        if tem == 0:
            tem = np.array([1, 0])
        else:
            tem = np.array([0, 1])
        
        tem = tem[np.newaxis, :] 
    
        inti = np.append(inti, tem, axis = 0)
        
    return inti
        
        
        

def read_and_decode(p):
    
    
    filename_queue = tf.train.string_input_producer([p]) 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   

    features = tf.parse_single_example(serialized_example,
                                       features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'img_raw' : tf.FixedLenFeature([], tf.string),
                                               })  

    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [128, 128])
    label = tf.cast(features['label'], tf.int32)
    
    print image.shape, label.shape
    
    
    
    return image, label


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

    epochs = 5    #15
    batch_size = 26
    learning_rate = 0.00008
    hidden = 4

    num = 100
    image_size = 128
    label_size = 2
#    ex = 2
    


    cwd = '/Users/wuzhenglin/Python_nice/SAL_LUNG/lung_hc_contrast_classification/contrast_lung_dataset.tfrecords' 
    im, lab = read_and_decode(cwd)
    img_batch, label_batch = tf.train.shuffle_batch([im, lab],
                                                batch_size = batch_size, capacity = num,
                                                min_after_dequeue = 16)
    
    cwd_ = '/Users/wuzhenglin/Python_nice/SAL_LUNG/lung_hc_contrast_classification/contrast_lung_dataset_test.tfrecords'
    im_, lab_ = read_and_decode(cwd_)  
    img_batch_, label_batch_ = tf.train.shuffle_batch([im_, lab_],
                                            batch_size = 15, capacity = 20,
                                            min_after_dequeue = 5)
    
#    train_loss = np.empty((num//(batch_size * ex)) * epochs)
#    train_acc = np.empty((num//(batch_size * ex)) * epochs)

    print ((num//(batch_size)) * epochs)
    train_loss = np.empty((num//(batch_size)) * epochs)
    train_acc = np.empty((num//(batch_size)) * epochs)

    x = tf.placeholder(tf.float32, shape = [None, image_size * image_size])
    y = tf.placeholder(tf.float32, shape = [None, label_size])

    weight_balance = tf.constant([0.1])
  
    X_train_ = tf.reshape(x, [-1, image_size, image_size, 1])

    #First layer
    W_conv1 = weight_variable([5, 5, 1, 4])
    b_conv1 = bias_variable([4])
      
    h_conv1 = tf.nn.relu(conv2d(X_train_, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    #Second layer
    W_conv2 = weight_variable([5, 5, 4, 8])
    b_conv2 = bias_variable([8])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
#    
##   Third layer
#    W_conv3 = weight_variable([5, 5, 8, 16])
#    b_conv3 = bias_variable([16])
#    
#    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
#    h_pool3 = max_pool_2x2(h_conv3)
    
    #Full connect layer
    W_fc1 = weight_variable([32 * 32 * 8, hidden])
    b_fc1 = bias_variable([hidden])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 32 * 32 * 8])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    #Output_Softmax
    
    W_fc2 = weight_variable([hidden, label_size])
    b_fc2 = bias_variable([label_size])
    
    
    out_feed = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2)
    y_conv = tf.nn.softmax(out_feed)
    
    print y_conv.shape
    
#    y_conv = np.squeeze(y_conv)


    #Train
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y, out_feed, weight_balance))
    optimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
   

    
    print 'Begin training'
    init_op = tf.initialize_all_variables()
   
    with tf.Session() as sess:
        
        sess.run(init_op)
        
        
        
        coord = tf.train.Coordinator()
        
        threads = tf.train.start_queue_runners(coord = coord)
        
        step = 1

        
        for ep in range(epochs):
        
            for i in range(num//batch_size):
                
                example, l = sess.run([img_batch, label_batch])
                
                ll = change(l)
                
                example = example.flatten()
                example = example.reshape([batch_size, image_size * image_size])
                
    
                
                
                
                
                feed_dict = {x: example, y: ll, keep_prob: 1.0}
                feed_dict_d = {x: example, y: ll, keep_prob: 0.5}
                
                sess.run(optimize, feed_dict = feed_dict_d)
                
                los, acc= sess.run([loss, accuracy], feed_dict = feed_dict)
                p = sess.run([y_conv], feed_dict = feed_dict)
                w2 = sess.run([W_fc1], feed_dict = feed_dict)
                train_loss[step -1] = los
                train_acc[step -1] = acc
            
                
                print "step: %d  loss: %.9f  accuracy: %.3f" % (step, los, acc)
                
#                print 'pred:\n', p
#                print 'label:\n', ll
                print W_fc1[0]

                
                step = step + 1
                
                
            
        
        print 'Finish Train.'
        coord.request_stop()
        coord.join(threads)
        
        plt.subplot(211)
        plt.plot(train_loss, 'r')
        plt.xlabel("epochs")
        plt.ylabel("Training loss")
        plt.grid(True)

        plt.subplot(212)
        plt.plot(train_acc, 'r')
        plt.xlabel("epochs")
        plt.ylabel('Training Accuracy')
        #plt.ylim(0.0, 1)
        plt.grid(True)
        plt.show()
 
        coord.request_stop()
        coord.join(threads)
        
        coord_ = tf.train.Coordinator()
        
        threads_ = tf.train.start_queue_runners(coord = coord_)
        

        print '2'
        print img_batch_.shape
        print label_batch_.shape

        
        example_, l_ = sess.run([img_batch_, label_batch_])
        print '3'
        ll_ = change(l_)
        print '4'       
        example_ = example_.flatten()
        example_ = example_.reshape([15, image_size * image_size])
        print '5'
        feed_dict_ = {x: example_, y: ll_, keep_prob: 1.0}
        print '6'
        acc_ = sess.run([accuracy], feed_dict = feed_dict_)
        fucker = sess.run([y_conv], feed_dict = feed_dict)
        
        print 'pred:\n', fucker
        print 'label:\n', ll_
        print '测试准确性：', acc_
        
        coord_.request_stop()
        coord_.join(threads_)    

if __name__ == '__main__':
    
    cnn_model()
