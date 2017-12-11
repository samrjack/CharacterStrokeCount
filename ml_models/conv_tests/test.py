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
import struct
from PIL import Image

image_size = 32
training_size = 128
n_inputs = 
n_out =

photos = ch.CharacterManager(image_size)

enc = OneHotEncoder()
enc.fit(np.array(list(range(0,34))).reshape(-1,1).tolist())

def pre_process(image):
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=.3)
    image = tf.image.random_contrast(image, .8, 1.2)

inputs = tf.placeholder(tf.float32, shape=[None, n_inputs])
y_ = tf.placeholder(tf.float32, shape=[None, n_out])

inputs = tf.map_fn(pre_process, inputs)

conv_1 = slim.conv2d(images, 32, [3,3], 1, padding='SAME')
max_pool_1 = tf.nn.max_pool2d(conv_1, [2,2], [2,2], padding='SAME')

conv_2 = tf.nn.conv2d(max_pool_1, 64, [3,3], padding='SAME')
max_pool_2 = tf.nn.max_pool2d(conv_2, [2,2], [2,2], padding='SAME')

flatten = tf.contrib.layers.flatten(max_pool_2)
out = tf.contrib.layers.fully_connected(flatten, n_out, activation_fn=None)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, y_))
train_op = tf.train.AdamOptimizer(learning_rate=.0001).minimize(loss, global_step=global_step)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(labels, 1)), tf.float32))

tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)
merged_summary_op = tf.summary.merge_all()

output_score = tf.nn.softmax(out)
predict_val_top3, predict_index_top3 = tf.nn.top_k(output_score, k=3)



with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=cord)
