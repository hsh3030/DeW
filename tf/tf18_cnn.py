import tensorflow as tf
import random
# import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

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

L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])