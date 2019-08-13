
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
# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
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
print(Y_train.shape)
print(Y_test.shape)

############################### hyperparameters #############################################
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
import numpy as np

def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28,), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(keep_prob)(x)
    prediction = Dense(10, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, output=prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
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
# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
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
print(Y_train.shape)
print(Y_test.shape)

############################### hyperparameters #############################################
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
import numpy as np
'''
def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(784,), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(keep_prob)(x)
    prediction = Dense(10, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, output=prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
'''
def build_network(keep_prob = 0.5, optimizer = 'adam'):
    inputs = Input(shape = (28, 28, 1), name = 'input')
    x = Conv2D(8, kernel_size = (3, 3), activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(keep_prob)(x)
    x1 = Conv2D(4, kernel_size = (3, 3), activation = 'relu', name = 'hidden2')(x)
    x1 = Dropout(keep_prob)(x1)
    x2 = Conv2D(2, kernel_size = (3, 3), activation = 'relu', name = 'hidden3')(x1)
    x2 = Dropout(keep_prob)(x2)
    x2 = Flatten()(x2)
    prediction = Dense(10, activation = 'softmax', name = 'output')(x2)
    model = Model(inputs = inputs, outputs = prediction)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model
    
def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size":batches, "optimizer":optimizers, "keep_prob":dropout}

from keras.wrappers.scikit_learn import KerasClassifier # classifier 분류
model = KerasClassifier(build_fn=build_network, verbose=1) # 사이킥런으로 랩핑을 하다.

hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV
# estimator=> model을 가져온다.
search = RandomizedSearchCV(estimator=model, param_distributions=hyperparameters, n_iter=10, n_jobs=1, cv=3, verbose=1)# 작업이 n_iter = 10회 수행, cv = 3겹 교차 검증

search.fit(X_train, Y_train)

print(search.best_params_)
