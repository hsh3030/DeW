import tensorflow as tf
import matplotlib.pyplot as plt
import random

tf.set_random_seed(777) # for reproducibility

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10
#############################################################
######  코딩하시오. X,Y,W,B, hypothesis, cost, train
#############################################################
# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0-9 digits recognition - 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])

W1 = tf.Variable(tf.random_normal([784, nb_classes]))
b1 = tf.Variable(tf.random_normal([nb_classes]))
layer1 = tf.nn.softmax(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([nb_classes, 100]))
b2 = tf.Variable(tf.random_normal([100]))
layer2 = tf.nn.softmax(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([100, 50]))
b3 = tf.Variable(tf.random_normal([50]))
layer3 = tf.nn.softmax(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_normal([50, 100]))
b4 = tf.Variable(tf.random_normal([100]))
layer4 = tf.nn.softmax(tf.matmul(layer3, W4) + b4)

W5 = tf.Variable(tf.random_normal([100, 50]))
b5 = tf.Variable(tf.random_normal([50]))
layer5 = tf.nn.softmax(tf.matmul(layer4, W5) + b5)

W6 = tf.Variable(tf.random_normal([50, 100]))
b6 = tf.Variable(tf.random_normal([100]))
layer6 = tf.nn.softmax(tf.matmul(layer5, W6) + b6)

W7 = tf.Variable(tf.random_normal([100, 20]))
b7 = tf.Variable(tf.random_normal([20]))
layer7 = tf.nn.softmax(tf.matmul(layer6, W7) + b7)

W8 = tf.Variable(tf.random_normal([20, 10]))
b8 = tf.Variable(tf.random_normal([10]))
layer8 = tf.nn.softmax(tf.matmul(layer7, W8) + b8)

W9 = tf.Variable(tf.random_normal([10, 10]))
b9 = tf.Variable(tf.random_normal([10]))
layer9 = tf.nn.softmax(tf.matmul(layer8, W9) + b9)

W10 = tf.Variable(tf.random_normal([10, nb_classes]))
b10 = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.nn.softmax(tf.matmul(layer9, W10) + b10)


cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
num_epochs = 15
batch_size = 100
# num_iterations = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(num_epochs):
        avg_cost = 0
        num_iterations = int(mnist.train.num_examples / batch_size)
                              # 55000 / 100 = 550
        for i in range(num_iterations): # 550
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) # 100
            cost_val, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})

            avg_cost += cost_val / num_iterations

        print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))
    print("Learning finished")

    # Test the model using test sets
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    # smaple image show and prediction
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))
    print("smaple image shape: ", mnist.test.images[r:r + 1].shape)
    plt.imshow(mnist.test.images[r:r + 1].reshape(28,28), cmap='Greys', interpolation='nearest')
    plt.show()
    