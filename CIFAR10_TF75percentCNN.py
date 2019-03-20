# -*- coding: utf-8 -*-
""" 
Author - Tejas Krishna Reddy
Date - 2nd Feb 2019

A Multi-Layer perceptron (Fully Connected Layer Network) to classify the objects from CIFAR image dataset

 Cifar-10 Classification using Keras Tutorial. The CIFAR-10 data set consists of 60000 32×32x3 color images in 10 classes,
 with 6000 images per class. There are 50000 training images and 10000 test images.
 
 Wrong answer
 With simple feed forward network an accuracy of 40+% was achieved. Now, using CNN, we get 70+% accuracy.
"""

import tensorflow as tf
import numpy as np
import keras


## Import data from keras
cifar10 = tf.keras.datasets.cifar10.load_data()

(x_train, y_train), (x_test, y_test) = cifar10
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)       # Convert classnumber into a one hot vector

# Preprocess training data - Normalising pixel values
x_train = x_train/255
x_test = x_test/255

# If dropout should exist or not
drop_out_flag = tf.placeholder(tf.bool)

# To normalize, subtract mean from all variables in xtrain and xtest
x_train -= np.mean(x_train)
x_test -= np.mean(x_test)


### Build the CNN architecture using tf backend.
# Placeholders for input and output
x = tf.placeholder(tf.float32, [None, 32 , 32 , 3])
y = tf.placeholder(tf.float32, [None, 10])

# Initializing Variables
global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')

#conv layer 1
conv1 = tf.layers.conv2d(inputs = x, filters = 32 ,kernel_size = [5,5], strides = 1,
                         padding = "same", activation = tf.nn.relu, name = 'conv1')
conv1_pool = tf.layers.max_pooling2d(inputs = conv1, pool_size= [5,5], strides = 1, padding = 'same')


#conv layer 2
conv2 = tf.layers.conv2d(inputs = conv1_pool, filters = 128, kernel_size = [5,5],
                         padding = "same", activation = tf.nn.relu, name = 'conv2')
conv2_pool = tf.layers.max_pooling2d(inputs = conv2, pool_size = [3,3], strides = 2 , padding = 'same')


#conv layer 3
conv3 = tf.layers.conv2d(inputs = conv2_pool, filters = 256, kernel_size = [5,5], 
                         padding = "same", activation = tf.nn.relu, name = 'conv3')
conv3_pool = tf.layers.max_pooling2d(inputs = conv3, pool_size = [3,3], strides = 2 , padding = 'same')

 
flat = tf.contrib.layers.flatten(conv3_pool)  # Flatten the Convoluted results and pass them through a fullyconnected layers

#Fully Connected Layer 1
fc1 = tf.layers.dense(flat,2048,activation = tf.nn.relu)
#Fully Connected Layer 2
fc2 = tf.layers.dense(fc1,1024,activation = tf.nn.relu)


#Softmax Layer
softmax = tf.layers.dense(fc2, units=10, activation = tf.nn.softmax)
y_pred_cls = tf.argmax(softmax,axis=1)

#Loss and Optimization
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=softmax, labels=y))

learning_rate = 1e-1
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1).minimize(loss, global_step=global_step)  


# Calculating Accuracy
correct_prediction = tf.equal(y_pred_cls, tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Set Parameters
batch_size = 100

#num_examples = 50000                # 50,000 examples is creating memory issues.
num_examples = 20000
num_test = 10000
num_epochs = 20                     # Reaches 80% for 10 epochs and 85% at 15-20 epochs 

cum_acc = 0
count = 1

## Training Accuracy
for e in range(num_epochs):
    for s in range(int(num_examples/batch_size)):
        batch_xs = x_train[s*batch_size:(s+1)*batch_size]
        batch_ys = y_train[s*batch_size:(s+1)*batch_size]
        
        _, _, _,batch_acc = sess.run([global_step, optimizer, loss, accuracy], feed_dict ={x:batch_xs, y: batch_ys,drop_out_flag: True})

    print("epoch", "\t", e)
    print("\t", "Training Accuracy:  ", batch_acc)

        

count = 0
total_acc = 0
batch_size = 64

## Testing Accuracy
for e in range(num_epochs):
    for s in range(int(num_test/batch_size)):
        batch_xs = x_test[s*batch_size:(s+1)*batch_size]
        batch_ys = y_test[s*batch_size:(s+1)*batch_size]
        
        _, _, _,batch_acc = sess.run([global_step, optimizer, loss, accuracy], feed_dict ={x:batch_xs, y: batch_ys,drop_out_flag: True})

    print("epoch", "\t", e)
    print("\t", "Testing Accuracy:  ", batch_acc)




