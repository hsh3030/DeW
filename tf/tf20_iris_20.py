import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np
tf.set_random_seed(777)

iris_data = np.load("./iris2_data.npy")

print("iris_data:",iris_data.shape)

x_train = iris_data[:,:-1]

y_train = iris_data[:,[-1]]

print(x_train)
print(y_train)
print(x_train.shape, y_train.shape)

cnt = 1
def layer(input, output,uplayer,dropout=0,end=False):
    global cnt
    w = tf.get_variable("w%d"%(cnt),shape=[input, output],initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([output]))
    if ~end:
        layer = tf.nn.relu(tf.matmul(uplayer, w)+b)
    else: layer = tf.matmul(uplayer, w)+b

    if dropout != 0:
        layer = tf.nn.dropout(layer, keep_prob=dropout)
    cnt += 1
    return layer

X = tf.placeholder(tf.float32,[None, 4])
Y = tf.placeholder(tf.int32,[None, 1])
keep_prob = 0

Y_one_hot = tf.one_hot(Y, 3) # one-hot
print("one_hot:", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, 3])
print("reshape one_hot:", Y_one_hot)

L1 = tf.layers.dense(X, 100, activation = tf.nn.relu)
L2 = tf.layers.dense(L1, 20, activation = tf.nn.relu)
L3 = tf.layers.dense(L2, 10, activation = tf.nn.relu)
logits = tf.layers.dense(L3, 3, activation= tf.nn.relu)


hypothesis = tf.nn.softmax(logits)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=tf.stop_gradient([Y_one_hot])))
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
prediction = tf.argmax(hypothesis, 1)

correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    
    for step in range(5001):
        _, cost_val, acc_val = sess.run([train, cost, accuracy], feed_dict={X: x_train, Y: y_train})
        if step % 100 == 0:
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))

    a, pred = sess.run([accuracy, prediction], feed_dict={X: x_train,Y: y_train})
    for p, y in zip(pred, y_train.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
    print(a)
    writer = tf.summary.FileWriter('./board/sample_1', sess.graph)