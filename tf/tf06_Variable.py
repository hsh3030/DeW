#  랜덤값으로 변수 1개를 만들고, 변수의 내용을 출력
import tensorflow as tf
# tf.set_random_seed(777)

W = tf.Variable(tf.random_normal([1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")

print(W)

W = tf.Variable([0.3], tf.float32) # W = 0.3

# sess = tf.Session()
# # 변수 초기화 오퍼레이션을 초기화 = tf.global_variables_initializer()
# sess.run(tf.global_variables_initializer())
# # print("sess.run(W) : ", sess.run(W))
# print(sess.run(W))
# sess.close()

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# aaa = W.eval()
# print(aaa)
# sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = W.eval(session = sess)
print(aaa)
sess.close()