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
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import itertools as it

def neural_net():
    # Get file names for training data.
    training_data = du.get_file_names()
    training_data = [x for x in training_data if 10 < du.get_file_stroke_count(x) < 13]

    # Shuffle data so files are in a random order.
    random.seed(42)
    random.shuffle(training_data)

    enc = OneHotEncoder()
    enc.fit(np.array(list(range(1,34))).reshape(-1,1).tolist())

    all_y = np.array(list(map(du.get_file_stroke_count, training_data)))
    all_y = enc.transform(all_y.reshape(-1,1)).toarray().tolist()

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
                    training_data
                    , all_y
                    , test_size=0.1
                    , random_state=42)

    # Define model meta parameters
    learning_rate = 0.06
    num_epoch = 50000
    training_size = 10

    n_inputs = du.import_photo(training_data[0]).shape[0]
    n_outputs = 33
    nodes = [n_inputs, 100, 16, n_outputs]

    # Build tensorflow model.
    inputs = tf.placeholder(tf.float32, [None, n_inputs], name="inputs")
    labels = tf.placeholder(tf.float32, [None, n_outputs], name="labels")

    weights = []
    biases  = []
    for i in range(len(nodes) - 1): 
        weights.append(tf.Variable(tf.random_uniform([nodes[i], nodes[i + 1]], -.1, .1), name=("weights_" + str(i))))
        biases.append(tf.Variable(tf.random_uniform([nodes[i + 1]], -.1, .1), name=("bias_" + str(i))))

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
                    indices = random.sample(range(len(X_train)), training_size)
                    data = du.get_pictures([X_train[i] for i in indices])
                    results = np.array([y_train[i] for i in indices])

                    old_w = weights[-1].eval()
                    _, cost, prediction = sess.run([optimizer, loss, y[-1]], feed_dict={inputs: data, labels: results.reshape(-1,n_outputs)})

                    # pred = list(map(scale_guess, prediction.reshape(-1).tolist()))
                    # print(list(zip(pred,results.tolist())), "\n\n")
                    print(epoch, "/", num_epoch, "--", cost, "--", np.linalg.norm(old_w - weights[-1].eval()))

                    if epoch % 1000 == 0:
                        learning_rate /= 10
                
            data = du.get_pictures(X_test)
            predictions = sess.run(y[-1], feed_dict={inputs:data})

            predList = [round(x) for x in predictions.reshape(-1).tolist()]
            
            error = list(zip(y_test, predList))
            error.sort()
            percentCorrect = len([x for x in error if x == (1, 1) or x == (0, 0)])/float(len(error))


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
