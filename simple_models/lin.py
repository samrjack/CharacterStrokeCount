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
import itertools as it

def test_models():
	# Get file names for training data.
	training_data = du.get_file_names()

	# Shuffle data so files are in a random order.
	random.seed(42)
	random.shuffle(training_data)

	# Split into training and test sets
	X_train, X_test, y_train, y_test = train_test_split(
			training_data
			, du.get_files_stroke_counts(training_data)
			, test_size=0.2
			, random_state=42)

	lin_func = simple_linear(X_train[:1000])

	show_error(lin_func, X_test, y_test)
	#show_error(lambda x: k_nearest_neighbor(X_train, y_train, x, 2), X_test[0:3000], y_test[0:3000])

def show_error(function, X_test, y_test):
	error = y_test - list(map(function, X_test))

	error.sort()

	percentCorrect = len([x for x in error if x == 0])/float(len(error))

	values = [k for k, g in it.groupby(error) if -10 <= k <= 10]
	amount = list(map(len,[list(g) for k, g in it.groupby(error) if -10 <= k <= 10]))

	plt.close('all')
	plt.figure(figsize=(13,10))
	plt.bar(values, amount, align='center', alpha=0.5)
	plt.xticks(values, values)
	plt.ylabel("Number of errors")
	plt.xlabel("Number of strokes")
	plt.title("Amount of error")
	plt.show()


	print('Percent error =', percentCorrect * 100, '%')
	
def simple_linear(training_files):
	# setup w vector with a constant
	X = du.get_pictures(training_files)
	y = du.get_files_stroke_counts(training_files)

	W = np.matmul(np.linalg.pinv(X), y)

	return (lambda x: int(np.round(np.matmul(du.import_photo(x), W))))
	
	
def lin_tensor():
	learning_rate = 0.1
	num_epoch = 10
	training_size = 1000
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
	x = tf.placeholder(tf.float32, [None, 32*32], name="x")
	y = tf.placeholder(tf.float32, [None, 1], name="y")

	W = tf.Variable(tf.zeros([32*32, 1]))

	y_pred = tf.matmul(x, W, name="predictions")
	error = y_pred - y
	mse = tf.reduce_mean(tf.square(error), name="mse")
	gradients = 2/training_size * tf.matmul(tf.transpose(x), error)
	train_step = tf.assign(W, W - learning_rate * gradients)

	init = tf.global_variables_initializer()

	# Run tensorflow model.
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(num_epoch):
			# Pick random samples of x to train with
			indices = random.sample(range(len(X_train)), training_size)
			data = du.get_pictures([X_train[i] for i in indices])
			results = np.array([float(y_train[i]) for i in indices])
			print(str(epoch) + "/" + str(num_epoch))
			sess.run(train_step, feed_dict={x: data, y: results.reshape((len(results),1))})
			print(np.max(W.eval()))

		return [W.eval()]


def k_nearest_neighbor(X_train, y_train, test, k=1):
	data = du.import_photo(test)
	increment_value = 5000
	i = 0;
	error = []
	while i < len(X_train):
		imp_x = du.get_pictures(X_train[i:min(i+increment_value, len(X_train))])
		y = y_train[i:min(i+increment_value, len(X_train))]
		i += increment_value

		temp_error = (imp_x - data)**2
		temp_error = list(map(sum, temp_error))
		#temp_error = [sum((x - data)**2),y) for x,y in list(zip(imp_x, y_train))]
		temp_error = list(zip(temp_error, y))
		temp_error.sort()
		error.append(temp_error[0:k])
	
	error = [x for subError in error for x in subError]
	error.sort()
	error = error[0:k]
	return int(np.round(np.mean(list(map(lambda x: x[1], error)))))
	

if __name__ == "__main__":
	lin_tensor()
