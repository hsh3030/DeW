# Multi-variable linear regression 1
import tensorflow as tf
import numpy as np
tf.set_random_seed(777) # for reproducibility

xy = np.loadtxt('./DeW/data/data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

nb_classes = 7
# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 16])
Y = tf.placeholder(tf.int32, shape=[None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
print("one_hot: ", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshpae one_hot: ", Y_one_hot)

'''
one_hot:  Tensor("one_hot:0", shape=(?, 1, 7), dtype=float32)
reshpae one_hot:  Tensor("Reshape:0", shape=(?, 7), dtype=float32)
'''

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight') 
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
logits = tf.matmul(X,W) + b
hypothesis = tf.nn.softmax(logits)

# cost/loss function
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels = tf.stop_gradient([Y_one_hot]))
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Accuracy Computation
# True if hypothesis>0.5 else False
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initializes global variables in the graph
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y:y_data})

        if step % 100 == 0:
            _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], feed_dict={X: x_data, Y: y_data})
            print("Step: {:5} \t Loss:{:3f}\tAcc:{:.2%}".format(step, cost_val, acc_val))
    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    #y_data : (N,1) = flatten => (N,) matches pred.shape
    for p,y in zip(pred, y_data.flatten()):
        print("[{}]\tprediction: {} \tTRUE \tY: {}".format(p == int(y), p, int(y)))
        