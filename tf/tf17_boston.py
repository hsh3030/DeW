import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.model_selection import train_test_split
tf.set_random_seed(777)

boston_data = np.load("./DeW/boston_housing_x.npy")

print("boston_data:",boston_data.shape)

x_train = boston_data[:,:-1]

y_train = boston_data[:,[-1]]

print(x_train.shape, y_train.shape)

x_train, x_test, y_train, y_test = train_test_split(x_train,y_train,test_size=0.2)
print(x_train.shape, y_train.shape)

cnt = 1
def layer(input, output,uplayer,dropout=0,end=False):
    global cnt
    w = tf.get_variable("w%d"%(cnt),shape=[input, output],initializer=tf.constant_initializer())
    b = tf.Variable(tf.random_normal([output]))
    if ~end:
        layer = tf.nn.leaky_relu(tf.matmul(uplayer, w)+b)
    else: layer = tf.matmul(uplayer, w)+b

    if dropout != 0:
        layer = tf.nn.dropout(layer, keep_prob=dropout)
    cnt += 1
    return layer

X = tf.placeholder(tf.float32,[None, 12])
Y = tf.placeholder(tf.float32,[None, 1])
keep_prob = 0

l1 = layer(12,50,X)
l2 = layer(50,50,l1)
l3 = layer(50,10,l2)
l4 = layer(10,10,l3)
logits = layer(10,1,l4,end=True)

hypothesis = tf.nn.sigmoid(logits)
# hypothesis = tf.nn.softmax(logits)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32)) 

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    
    for step in range(5001):
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

from sklearn.metrics import mean_squared_error
def RMSE(y_test, pred): # y_test 와 y_predict 비교하기 위한 함수 (원래의 값과 예측값을 비교)
    return np.sqrt(mean_squared_error(y_test, pred)) # 비교하여 그 차이를 빼준다
print("RMSE : ", RMSE(y_test, pred))

# R2 구하기 (1에 가까울 수록 좋다.)
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, pred)
print("R2 : ", r2_y_predict)
