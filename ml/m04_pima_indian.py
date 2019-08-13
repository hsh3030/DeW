from keras.models import Sequential
from keras.layers import Dense
import numpy
import tensorflow as tf
import os
import matplotlib.pyplot as plt
# seed 값 생성
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드

print(os.getcwd())

dataset = numpy.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:, 0:8]
Y = dataset[:, 8]

# model 설정
model = Sequential()
model.add(Dense(64, input_dim=8, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
# 0,1의 데이터가 있을때 가장 많이 쓰이는 activation = sigmoid(2진 모드) 로 분류한다.
model.add(Dense(1, activation='sigmoid'))

# model complie
# loss='binary_crossentropy' => sigmoid일때 쓴다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model 실행
model.fit(X, Y, epochs=2000, batch_size=10)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))

