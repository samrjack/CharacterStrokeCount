import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
		inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import data_statistics.data_util as du

import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def linear_regression():
	# Get file names for training data.
	training_data = du.get_file_names()

	# Shuffle data so files are in a random order.
	random.seed(42)
	random.shuffle(training_data)

	# Split into training and test sets
	X_train, X_test, y_train, y_test = train_test_split(
			training_data
			, map(du.get_file_stroke_count)
			, test_size=0.2
			, random_state=42)

	# setup w vector with a constant
	w = np.zeros((1 + 32*32,1))
	# Gradiant decent
	error = []
	epochs = 100
	training_constant = .1
	for epoch in range(epochs):
		indices = random.sample(range(len(X_train)), 1000)
		data = du.get_pictures([X_train[i] for i in indices])
		results = np.array([y_train[i] for i in indices])
		with_constant = np.vstack(np.ones(data.shape[1]), data)
		guess = np.dot(with_constant, w)
		err = np.subtract(results, np.mean(guess, axis=4))
		error.append(np.mean(np.abs(err)))
		w += training_constant * guess

	
def lin_tensor():
	# Get file names for training data.
	training_data = du.get_file_names()

	# Shuffle data so files are in a random order.
	random.seed(42)
	random.shuffle(training_data)

	# Split into training and test sets
	X_train, X_test, y_train, y_test = train_test_split(
			training_data
			, list(map(du.get_file_stroke_count, training_data))
			, test_size=0.1
			, random_state=42)

	# Build tensorflow model.
	x = tf.placeholder(tf.float32, [None, 32*32])

	W = tf.Variable(tf.zeros([32*32, 1]))
	b = tf.Variable(tf.zeros([1]))

	y = tf.matmul(x, W) + b
	y_ = tf.placeholder(tf.float32, [None, 1])

	LeastSquares = tf.reduce_mean((y - y_)**2)

	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(LeastSquares)

	init = tf.global_variables_initializer()

	# Run tensorflow model.
	error = []
	num_epoch = 10
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(num_epoch):
			indices = random.sample(range(len(X_train)), 1000)
			data = du.get_pictures([X_train[i] for i in indices])
			results = np.array([float(y_train[i]) for i in indices])
			print(str(epoch) + "/" + str(num_epoch))
			sess.run(train_step, feed_dict={x: data, y_: results.reshape((len(results),1))})
			print(np.max(W.eval()))

		return [W.eval(), b.eval()]
