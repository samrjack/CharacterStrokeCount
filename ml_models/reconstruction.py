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

def reconstruction():
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

    # Define model meta parameters
    learning_rate = 0.0001
    num_epoch = 5000
    training_size = 10

    n_inputs = du.import_photo(training_data[0]).shape[0]
    n_outputs = n_inputs
    n_hidden_1 = 200
    
    nodes = [5, 10, 20, 40, 60, 80, 100, 130, 150, 180, 200, 240, 280, 300, 350, 400, 450, 500, 550, 600, 700, 800, 900, 1000, n_inputs]

    node = [5, 10]
    error = []

    for n_hidden_1 in nodes:
        learning_rate = 0.0001

        # Build tensorflow model.
        inputs = tf.placeholder(tf.float32, [None, n_inputs], name="inputs")
        labels = tf.placeholder(tf.float32, [None, n_outputs], name="labels")

        weight1 = tf.Variable(tf.random_uniform([n_inputs, n_hidden_1], -.1, .1), name="weight1")
        biases1 = tf.Variable(tf.random_uniform([n_hidden_1], -.1, .1), name="bias1")

        weight2 = tf.Variable(tf.random_uniform([n_hidden_1, n_outputs], -.1, .1), name="weight2")
        biases2 = tf.Variable(tf.random_uniform([n_outputs], -.1, .1), name="bias2")


        u1 = tf.add(tf.matmul(inputs, weight1), biases1)
        y1 = tf.nn.sigmoid(u1)
        # y1 = tf.nn.relu_layer(x=inputs, weights=weight1, biases=biases1)

        u2 = tf.add(tf.matmul(u1, weight2), biases2)
        output = tf.nn.sigmoid(u2)

        loss = tf.nn.l2_loss(output - inputs)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        init = tf.global_variables_initializer()
        
        # Run tensorflow model.
        with tf.Session() as sess:
                sess.run(init)
                for epoch in range(num_epoch):
                        # Pick random samples of x to train with
                        indices = random.sample(range(len(X_train)), training_size)

                        data = du.get_pictures([X_train[i] for i in indices])
                        results = np.array([float(y_train[i]) for i in indices])

                        _, cost, prediction = sess.run([optimizer, loss, output], feed_dict={inputs: data, labels: data}) 
                        print(str(epoch) + "/" + str(num_epoch) + " -- " + str(n_hidden_1) + "   " + str(cost))
                        if(epoch % 250 == 0):
                            learning_rate /= 10

                data = du.get_pictures(X_test)
                cost = sess.run(loss, feed_dict={inputs:data})

                error.append(cost)

    plt.plot(nodes, error)
    plt.show()
             
if __name__ == "__main__":
    reconstruction()
