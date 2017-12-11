import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
                inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import data_statistics.char_handler as ch

import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import itertools as it

def neural_net():

    image_size = 32
    photos = ch.CharacterManager(image_size)

    enc = OneHotEncoder()
    enc.fit(np.array(list(range(1,34))).reshape(-1,1).tolist())

    # Define model meta parameters
    learning_rate = 0.1
    num_epoch = 5000000
    training_size = 10

    n_inputs = image_size**2
    n_outputs = enc.n_values_.tolist()[0] - 1
    nodes = [n_inputs, 100, 16, n_outputs]

    # Build tensorflow model.
    inputs = tf.placeholder(tf.float32, [None, n_inputs], name="inputs")
    labels = tf.placeholder(tf.float32, [None, n_outputs], name="labels")

    weights = []
    biases  = []
    for i in range(len(nodes) - 1): 
        weights.append(tf.Variable(tf.random_uniform([nodes[i], nodes[i + 1]], -.5, .5), name=("weights_" + str(i))))
        biases.append(tf.Variable(tf.random_uniform([nodes[i + 1]], -.5, .5), name=("bias_" + str(i))))

    u = [tf.add(tf.matmul(inputs, weights[0]), biases[0])]
    y = [tf.nn.sigmoid(x=u[0])]

    for i in range(1,len(nodes) - 1):
        u.append(tf.add(tf.matmul(y[i-1], weights[i]), biases[i]))
        y.append(tf.nn.sigmoid(x=u[i]))

    loss = tf.nn.l2_loss(y[-1] - labels)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    # Run tensorflow model.
    with tf.Session() as sess:
            sess.run(init)
            for epoch in range(num_epoch):
                    # Pick random samples of x to train with
                    batch = photos.training_batch(50)
                    batch[1] = enc.transform(batch[1]).toarray()

                    old_w = weights[-1].eval()
                    _, cost, prediction = sess.run([optimizer, loss, y[-1]], feed_dict={inputs: batch[0], labels: batch[1]})

                    # pred = list(map(scale_guess, prediction.reshape(-1).tolist()))
                    # print(list(zip(pred,results.tolist())), "\n\n")
                    if epoch % 1000 == 0:
                        print(epoch, "/", num_epoch, "--", cost, "--", np.linalg.norm(old_w - weights[-1].eval()))
                        learning_rate /= 3
                
            batch = photos.testing_data()
            predictions = sess.run(y[-1], feed_dict={inputs:batch[0]})

            predList = [list(map(round, x)) for x in predictions.tolist()]
            
            error = list(zip(enc.transform(batch[1]).toarray().tolist(), predList))
            error.sort()
            percentCorrect = len([x for x in error if x[0] == x[1]])/float(len(error))


            print('Percent correct =', percentCorrect * 100, '%')

            for k, g in it.groupby(error):
                print(k, len(list(g)))
#            values = [k for k, g in it.groupby(error)]
#            amount = list(map(len, [list(g) for k, g in it.groupby(error)]))
#            plt.close('all')
#            plt.figure(figsize=(13,10))
#            plt.bar(values, amount, align='center', alpha=0.5)
#            plt.xticks(values, values)
#            plt.ylabel("Number of errors")
#            plt.xlabel("Number of strokes")
#            plt.title("Amount of error")
#            plt.show()


if __name__ == "__main__":
    neural_net()
