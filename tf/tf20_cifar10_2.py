from keras.datasets import cifar10
import tensorflow as tf
import random
from keras.datasets import cifar10
import numpy as np
# import matplotlib.pyplot as plt

def next_batch(num, data, labels):

    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

tf.set_random_seed(777)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

nb_classse = 10

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.one_hot(Y, nb_classse)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classse])

learning_rate = 0.01
training_epochs = 20
batch_size = 100

W1 = tf.Variable(tf.random_normal([32, 32, 3, 32], stddev=0.01))
print(W1)
L4 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') 
print('L4 : ', L4) 
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 
print('L4 : ', L4)

L1 = tf.layers.conv2d(L4, 64, [3, 3], activation=tf.nn.relu, padding = 'SAME')
L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])
L1 = tf.layers.dropout(L1, 0.5)

L3 = tf.layers.conv2d(L1, 128, [3, 3], activation=tf.nn.relu, padding = 'SAME')
L3 = tf.layers.max_pooling2d(L3, [2, 2], [2, 2])
L3 = tf.layers.dropout(L3, 0.8)

L2 = tf.contrib.layers.flatten(L3)
L2 = tf.layers.dense(L2, 128, activation=tf.nn.relu)
L2 = tf.layers.dropout(L2, 0.3)

L3 = tf.layers.flatten(L2)
logits = tf.layers.dense(L3, 10, activation=tf.nn.relu)

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=tf.stop_gradient([Y_one_hot])))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# train my model
print('Learning started. It takes sometime.')
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = x_train.shape[0] // batch_size
        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(batch_size, x_train, y_train)
            feed_dict = {X: batch_xs, Y: batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch
        print('Epoch: ', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))
    print('Learning Finished!')
    
    test_accuracy = 0.0  
    for i in range(10):
        test_batch = next_batch(1000, x_test, y_test)
        test_accuracy = test_accuracy + accuracy.eval(feed_dict={X: batch_xs, Y: batch_ys})
    test_accuracy = test_accuracy / 10;
    print("테스트 데이터 정확도: %f" % test_accuracy)