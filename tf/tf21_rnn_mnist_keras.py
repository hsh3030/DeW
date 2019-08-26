#-*- coding: utf-8 -*-

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, LSTM

import numpy
import os
import tensorflow as tf

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28).astype('float32') / 255
print(Y_train.shape) # (60000,) 벡터는 연결된 60000개다, scaler는 데이터1개
print(Y_test.shape)

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(28,28)))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(10, activation = "softmax"))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

early_stoping_callback = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs= 100, batch_size= 100)

print("\n test acc: %.4f"%(model.evaluate(X_test, Y_test)[1]))