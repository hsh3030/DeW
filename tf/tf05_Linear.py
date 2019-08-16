import tensorflow as tf
# 난수표에서 777번째 표를 사용한다.
tf.set_random_seed(777)

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# Variable 변수 => 
# tf.random_normal => random 하게 값을 넣는다.
W = tf.Variable(tf.random_normal([1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")

hypothesis = x_train * W + b
############################## model.compile #######################################################
# cost/loss function
# tf.reduce_mean => 평균을 낸다. tf.square (== loss = 'mse' ) => 제곱
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# optimizer
# GradientDescentOptimizer = 경사하강법
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)
###################################################################################################

#################################### model.fit // evaluate #######################################
# Launch the grape in a session
# with 문은 with 블록이 끝나면 자동으로 close 해준다.
# with는 블록단위의 프로세스의 시작과 끝에 대한 처리를 해준다.
# cost = loss 0에 근접한 모델을 찾는것
with tf.Session() as sess:
    # Initializes global cariables in the graph
    # global_variables_initializer => tensorflow는 변수를 만들면 항상 초기화 시켜야 한다.
    sess.run(tf.global_variables_initializer()) 

    # Fit the line
    # 2001 => epochs 값
    # sess.run => fit 값
    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])

        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)
#####################################################################################################

