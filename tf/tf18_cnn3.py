import tensorflow as tf
import random
# import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# hyper parameters
learning_rate = 0.001
training_epochs = 50
batch_size = 10

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1]) # img 28 X 28 X 1 (black/white)
Y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape = (?, 28, 28, 1)
# 3, 3 은 kernel_size 몇 * 몇으로 자를것인지 정하며 1은 그레이 색상을 표현 32는 output
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01)) 
print('W1: ', W1)
# Conv -> (?, 28, 28, 1)
# Pool -> (?, 14, 14, 1)

# [1, 1, 1, 1] 양 끝 1은 default
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME') # 26 * 26 , 32장 -> padding = same로 인해 28 * 28 , 32장
print('L1 : ', L1) # ("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # 2, 2  2 * 2로 자른걸 2칸씩 이동
print('L1 : ', L1) # ("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)

# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)) # 32는 w1 output가 들어온 것이다.
print('W2: ', W2)

# Conv -> (?, 14, 14, 64)
# Pool -> (?, 7, 7, 64)

L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
print('L2 : ', L2)
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print('L2 : ', L2)

W3 = tf.Variable(tf.random_normal([3, 3, 32, 64]))

L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
print('L3 : ', L3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print('L3 : ', L3)

W4 = tf.Variable(tf.random_normal([2, 2, 32, 64]))

L4 = tf.nn.conv2d(L3, W4, strides =[1, 2, 2, 1], padding = 'SAME')
print('L4 : ', L4)
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print('L4 : ', L4)

L4_flat = tf.reshape(L4, [-1, 1 * 1 * 64])

################ 8/23 시작 ################
# Final FC 7X7X64 input -> 10 outputs
W5 = tf.get_variable("W5", shape=[1 * 1 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L4_flat, W5) + b

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
print('Learning started. It takes sometime.')

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch: ', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy: ', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1]}))
