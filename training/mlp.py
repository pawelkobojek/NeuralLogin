import tensorflow as tf
import numpy as np
import sys

print sys.argv

filename = "../training_data/" + sys.argv[1] + ".txt"
with open(filename) as f:
    dataset = [d.split(',') for d in f.read().splitlines()]
    labels = [int(c[0]) for c in dataset]
    seqs = [map(lambda x: int(x), c[2:33]) for c in dataset]

labels = np.array([labels])
seqs = np.array(seqs)

sequence_len = len(seqs[0])

x  = tf.placeholder("float", [None, sequence_len])
y_ = tf.placeholder("float", [None, len(labels[0])])

W1 = tf.Variable(tf.random_uniform([sequence_len, 35], -1.0, 1.0))
b1 = tf.Variable(tf.zeros([35]))
h1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.random_uniform([35, 15], -1.0, 1.0))
b2 = tf.Variable(tf.zeros([15]))
h2 = tf.nn.sigmoid(tf.matmul(h1, W2) + b2)

W3 = tf.Variable(tf.random_uniform([15, 1], -1.0, 1.0))
b3 = tf.Variable(tf.zeros([1]))
y = tf.nn.sigmoid(tf.matmul(h2, W3) + b3)

loss  = tf.reduce_mean(tf.square(y_ - y))
train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

accuracy = tf.reduce_mean(tf.abs(y - y_))

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

for i in xrange(20000):
    sess.run(train, feed_dict={x: seqs, y_: labels})
    if i % 100 == 0:
        print sess.run(W1)
        train_accuracy = accuracy.eval(feed_dict={x: seqs, y_: labels})
        print "step %d, training accuracy %g"%(i, train_accuracy)
