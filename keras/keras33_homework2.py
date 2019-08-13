#-*- coding: utf-8 -*-

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy
import os
import tensorflow as tf

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# matplotlib <= 데이터 시각화
# import matplotlib.pyplot as plt
# digit = X_train[4825] # mnist image 주소 지정
# plt.imshow(digit, cmap = plt.cm.binary)
# plt.show()


# astype / 255 -> 0,1로 데이터 전처리 (minmax scaler) 0~255개의 데이터를 가지고 있기 때문에
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
print(Y_train.shape) # (60000,) 벡터는 연결된 60000개다, scaler는 데이터1개
print(Y_test.shape)
# np_utils.to_categorical에 넣으면 분류값의 분류가 된다. -> # (60000, 10) 10은 데이터의 갯수 <= onehot encoding
# 원-핫 인코딩은 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 
# 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식입니다. 이렇게 표현된 벡터를 원-핫 벡터(One-hot vector)라고 합니다. 
'''
# one-hot(원핫)인코딩이란? 단 하나의 값만 True이고 나머지는 모두 False인 인코딩을 말한다. 
# 즉, 1개만 Hot(True)이고 나머지는 Cold(False)이다. 
# 예를들면 [0, 0, 0, 0, 1]이다. 5번째(Zero-based 인덱스이므로 4)만 1이고 나머지는 0이다
'''
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# 컨볼루션 신경망의 설정
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), input_shape = (28, 28, 1), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2)) # pool_size=2 = (2,2)
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax')) # 분류모델 마지막은 무조건 'softmax'를 쓴다/.

#loss='categorical_crossentropy' 분류모델(지정된 output값만 출력되게 하는 모델)에서 쓴다.
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) 

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
# model.summary()

history = model.fit(X_train, Y_train, validation_data=(X_test,Y_test), 
                    epochs= 30, batch_size=10, verbose=1,
                    callbacks=[early_stopping_callback])
# 분류모델일 때는 accuracy로 적용한다.
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))
