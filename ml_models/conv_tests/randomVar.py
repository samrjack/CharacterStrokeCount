import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
                inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir) 

import data_statistics.char_handler as ch

import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools as it

image_size = 32
save_file_name = "saved_models/model_out3.ckpt"
confusion_file = "confusion_matrix3.txt"
photos = ch.CharacterManager(image_size)

enc = OneHotEncoder()
enc.fit(np.array(list(range(1,34))).reshape(-1,1).tolist())

training_size = 60

num_itter = 100000

n_inputs = image_size**2
n_outputs = enc.n_values_.tolist()[0] - 1


inputs = tf.placeholder(tf.float32, shape=[None, n_inputs])
y_ = tf.placeholder(tf.float32, shape=[None, n_outputs])

x = tf.reshape(inputs, [-1, image_size, image_size, 1])
x = tf.image.random_brightness(x, max_delta=0.2)
x = tf.image.random_contrast(x, .8, 1.2)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def max_pool_1x1(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

conv_size = 5
depth = [15, 15, 20]
full_connect = 100
W_conv0 = weight_variable([conv_size, conv_size, 1, depth[0]])
b_conv0 = bias_variable([depth[0]])

W_conv1 = weight_variable([conv_size, conv_size, depth[0], depth[1]])
b_conv1 = bias_variable([depth[1]])

W_conv2 = weight_variable([conv_size, conv_size, depth[1], depth[2]])
b_conv2 = bias_variable([depth[2]])

h_conv0 = tf.nn.relu(conv2d(x, W_conv0) + b_conv0)
h_pool0 = max_pool_1x1(h_conv0)

h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

final_size = int(image_size/8)

W_fc1 = weight_variable([final_size * final_size * depth[2], full_connect])
b_fc1 = bias_variable([full_connect])

h_pool2_flat = tf.reshape(h_pool2, [-1, final_size * final_size * depth[2]])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([full_connect, n_outputs])
b_fc2 = bias_variable([n_outputs])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
prediction = tf.argmax(y_conv, 1)
correct = tf.argmax(y_, 1)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    try:
        save.restore(sess, save_file_name)
    except:
        sess.run(tf.global_variables_initializer())
    for i in range(num_itter):
        batch = photos.training_batch(50)
        batch[1] = enc.transform(batch[1]).toarray()
        if i % 100 == 0:
            final_batch = photos.testing_data()
            train_accuracy = accuracy.eval(feed_dict={
                inputs: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            print('test accuracy %g' % accuracy.eval(feed_dict={
                inputs: final_batch[0], y_: enc.transform(final_batch[1]).toarray(), keep_prob: 1.0}))
#            f = open(confusion_file, 'w')
#            f.write(str(confusion_matrix(y_true=correct, y_pred=y_pred)))
#            f.close()
        train_step.run(feed_dict={inputs: batch[0], y_: batch[1], keep_prob: 0.5})

        if i % 1000 == 999:
            save_path = saver.save(sess, save_file_name)

    # Training Complete

    batch = photos.testing_data()
    print('test accuracy %g' % accuracy.eval(feed_dict={
        inputs: batch[0], y_: enc.transform(batch[1]).toarray(), keep_prob: 1.0}))
    y_pred = prediction.eval(feed_dict={inputs: batch[0], keep_prob: 1.0})
    correct = correct.eval({y_: enc.transform(batch[1]).toarray()})
    print(y_pred)
    print(correct)
    
    f = open(confusion_file, 'w')
    f.write(str(confusion_matrix(y_true=correct, y_pred=y_pred)))
    f.close()
