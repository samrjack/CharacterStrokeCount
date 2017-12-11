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
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected

image_size = 32
save_file_name = "saved_models/model_out3.ckpt"
confusion_file = "confusion_matrix3.txt"
photos = ch.CharacterManager(image_size)

enc = OneHotEncoder()
enc.fit(np.array(list(range(1,34))).reshape(-1,1).tolist())

training_size = 60

num_itter = 35000

n_inputs = image_size**2
n_outputs = enc.n_values_.tolist()[0] - 1


inputs = tf.placeholder(tf.float32, shape=[None, n_inputs])
y_ = tf.placeholder(tf.float32, shape=[None, n_outputs])

x = tf.image.random_brightness(inputs, max_delta=0.2)
x_image = tf.image.random_contrast(tf.reshape(x, [-1, image_size, image_size, 1]), .8, 1.2)

conv_net = conv_2d(conv_net, 32, 2, activation='relu')
conv_net = max_pool_2d(conv_net, 2)
conv_net = conv_2d(conv_net, 64, 2, activation='relu')
conv_net = max_pool_2d(conv_net, 2)
conv_net = fully_connected(conv_net, 1024, activation='relu')
conv_net = dropout(conv_net, .8)
conv_net = fully_connected(conv_net, 10, activation='softmax')
conv_net = regression(conv_net, optimizer='adam', loss='categorical_crossentropy', name='output')

model = tflearn.DNN(conv_net)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_itter):
        batch = photos.training_batch(50)
        batch[1] = enc.transform(batch[1]).toarray()
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                inputs: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
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
