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

def neural_net():
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

        scale_result = lambda y: y/100.0
        scale_guess  = lambda _y: round(_y * 100)

        # Define model meta parameters
        learning_rate = 0.00006
        num_epoch = 1000
        training_size = 100

        n_inputs = du.import_photo(training_data[0]).shape[0]
        n_outputs = 1
        n_hidden_1 = 100
        n_hidden_2 = 200
        n_hidden_3 = 50

        # Build tensorflow model.
        inputs = tf.placeholder(tf.float32, [None, n_inputs], name="inputs")
        labels = tf.placeholder(tf.float32, [None, n_outputs], name="labels")

        weight1 = tf.Variable(tf.random_uniform([n_inputs, n_hidden_1], -.1, .1), name="weight1")
        biases1 = tf.Variable(tf.random_uniform([n_hidden_1], -.1, .1), name="bias1")

        weight2 = tf.Variable(tf.random_uniform([n_hidden_1, n_hidden_2], -.1, .1), name="weight2")
        biases2 = tf.Variable(tf.random_uniform([n_hidden_2], -.1, .1), name="bias2")

        weight3 = tf.Variable(tf.random_uniform([n_hidden_2, n_hidden_3], -.1, .1), name="weight3")
        biases3 = tf.Variable(tf.random_uniform([n_hidden_3], -.1, .1), name="bias3")

        weight4 = tf.Variable(tf.random_uniform([n_hidden_3, n_outputs], -.1, .1), name="weight4")
        biases4 = tf.Variable(tf.random_uniform([n_outputs], -.1, .1), name="bias4")

        u1 = tf.add(tf.matmul(inputs, weight1), biases1)
        # y1 = tf.nn.sigmoid(u1)
        y1 = tf.nn.relu_layer(x=inputs, weights=weight1, biases=biases1)

        u2 = tf.add(tf.matmul(u1, weight2), biases2)
        # y2 = tf.nn.sigmoid(u2)
        y2 = tf.nn.relu_layer(x=y1, weights=weight2, biases=biases2)

        u3 = tf.add(tf.matmul(u2, weight3), biases3)
        # y3 = tf.nn.sigmoid(u3)
        y3 = tf.nn.relu_layer(x=y2, weights=weight3, biases=biases3)

        u4 = tf.add(tf.matmul(u3, weight4), biases4)
        output = tf.nn.sigmoid(u4)

        loss = tf.nn.l2_loss(output - scale_result(labels))
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

                        _, cost, prediction = sess.run([optimizer, loss, output], feed_dict={inputs: data, labels: results.reshape(-1,n_outputs)})

                        weight1.eval()
                        weight2.eval()
                        weight3.eval()

                        biases1.eval()
                        biases2.eval()
                        biases3.eval()

                        # pred = list(map(scale_guess, prediction.reshape(-1).tolist()))
                        # print(list(zip(pred,results.tolist())), "\n\n")
                        print(str(epoch) + "/" + str(num_epoch) + " -- " + str(cost))

                data = du.get_pictures(X_test)
                predictions = sess.run(output, feed_dict={inputs:data})

                predList = [round(scale_guess(x)) for x in predictions.reshape(-1).tolist()]
                
                error = (np.array(y_test) - np.array(predList)).tolist()
                error.sort()
                percentCorrect = len([x for x in error if x == 0])/float(len(error))

                values = [k for k, g in it.groupby(error)]
                amount = list(map(len, [list(g) for k, g in it.groupby(error)]))

                print('Percent correct =', percentCorrect * 100, '%')

                plt.close('all')
                plt.figure(figsize=(13,10))
                plt.bar(values, amount, align='center', alpha=0.5)
                plt.xticks(values, values)
                plt.ylabel("Number of errors")
                plt.xlabel("Number of strokes")
                plt.title("Amount of error")
                plt.show()


if __name__ == "__main__":
    neural_net()
