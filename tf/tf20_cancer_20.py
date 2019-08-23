import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.model_selection import train_test_split
tf.set_random_seed(66)

cancer_data = np.load("./cancer_data.npy")

print("cancer_data:",cancer_data.shape)

x_train = cancer_data[:,:-1]

y_train = cancer_data[:,[-1]]

print(x_train.shape, y_train.shape)


X = tf.placeholder(tf.float32,[None, 30])
Y = tf.placeholder(tf.float32,[None, 1])
keep_prob = 0
x_train, x_test, y_train, y_test = train_test_split(x_train,y_train, test_size=0.2)
print(x_train.shape, y_train.shape)
L1 = tf.layers.dense(X, 50, activation = tf.nn.relu)
L2 = tf.layers.dense(L1, 20, activation = tf.nn.relu)
L3 = tf.layers.dense(L2, 10, activation = tf.nn.relu)
logits = tf.layers.dense(L3, 1, activation= tf.nn.relu)

hypothesis = tf.nn.sigmoid(logits)
# hypothesis = tf.nn.softmax(logits)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
 
    sess.run(tf.global_variables_initializer())
    
    for step in range(50001):
        _, cost_val, acc_val = sess.run([train, cost, accuracy], feed_dict={X: x_train, Y: y_train})
        if step % 100 == 0:
            print("Step: {:5}\tCost: {:f}\tAcc: {:.2%}".format(step, cost_val, acc_val))

    # Let's see if we can predict
    a, pred = sess.run([accuracy, predicted], feed_dict={X: x_test,Y: y_test})
    # y_data: (N, 1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_train.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
    print(a)
    writer = tf.summary.FileWriter('./board/sample_1', sess.graph)