import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype = np.float32)

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

#================================ 한개의 layer ===================================
W1 = tf.Variable(tf.random_normal([2,100], name='weight')) # (input, output)
b1 = tf.Variable(tf.random_normal([100]), name='bias') # output
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1) # activation함수
#=================================================================================
W2 = tf.Variable(tf.random_normal([100,50], name='weight'))
b2 = tf.Variable(tf.random_normal([50]), name='bias')
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([50,100], name='weight'))
b3 = tf.Variable(tf.random_normal([100]), name='bias')
layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_normal([100,75], name='weight'))
b4 = tf.Variable(tf.random_normal([75]), name='bias')
layer4 = tf.nn.relu(tf.matmul(layer3, W4) + b4)

W5 = tf.Variable(tf.random_normal([75,100], name='weight'))
b5 = tf.Variable(tf.random_normal([100]), name='bias')
layer5 = tf.sigmoid(tf.matmul(layer4, W5) + b5)

W6 = tf.Variable(tf.random_normal([100,100], name='weight'))
b6 = tf.Variable(tf.random_normal([100]), name='bias')
layer6 = tf.sigmoid(tf.matmul(layer5, W6) + b6)

W7 = tf.Variable(tf.random_normal([100,100], name='weight'))
b7 = tf.Variable(tf.random_normal([100]), name='bias')
layer7 = tf.sigmoid(tf.matmul(layer6, W7) + b7)

W8 = tf.Variable(tf.random_normal([100,100], name='weight'))
b8 = tf.Variable(tf.random_normal([100]), name='bias')
layer8 = tf.sigmoid(tf.matmul(layer7, W8) + b8)

W9 = tf.Variable(tf.random_normal([100,100], name='weight'))
b9 = tf.Variable(tf.random_normal([100]), name='bias')
layer9 = tf.sigmoid(tf.matmul(layer8, W9) + b9)

W10 = tf.Variable(tf.random_normal([100,1], name='weight'))
b10 = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = tf.sigmoid(tf.matmul(layer9, W10) + b10)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        _, cost_val, w_val = sess.run([train, cost, W10], feed_dict={X: x_data, Y: y_data})
        
        if step % 100 == 0:
            print(step, cost_val, w_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\Correct: ", c, "\nAccuracy: ", a)