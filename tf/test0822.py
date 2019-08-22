import os
import pandas as pd 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, MaxPooling1D
from keras.layers import LSTM, Flatten
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt 
import math
from sklearn.metrics import mean_squared_error
import io
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D
from keras.layers import Embedding
test_data = pd.read_csv("./data/test0822_hsh.csv", encoding='utf-8', names=['1', '2', '3', '6', '9', '12', '15', '18', 'end'])

print(test_data)
print(test_data.shape) # (5480, 9)
print(type(test_data))
print(test_data.info())


# 레이블로 자른다.
y = test_data.loc[:, ['2', '3', '6', '9', '12', '15', '18', 'end']]
x = test_data.loc[:, ['2', '3', '6', '9', '12', '15', '18']]

print(x.shape) # (5480, 7)
print(y.shape) # (5480,)

x = np.asarray(x)
y = np.asarray(y)
print(x.shape)
print(y.shape)
from sklearn.model_selection import train_test_split  

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=88, test_size=0.2, shuffle = True)
scaler = MinMaxScaler()
#scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
print(x_train.shape) #(4375, 1, 7)
print(x_test.shape) #(1094, 1, 7)

model = Sequential()
model.add(LSTM(128, input_shape = (1, 7)))
# model.add(Conv1D(475,input_shape=(5,), activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.compile(loss='mse', optimizer='adam')
model.summary()


model.compile(loss='mse', optimizer = 'adam', metrics=['accuracy']) 
model.fit(x_train, y_train, batch_size = 100, nb_epoch = 3000, validation_split = 0.05)

y_predict = model.predict(x_test)
print("y_predict : ", y_predict)

# RMSE 구하기 (RMSE: 낮을수록 좋다.)
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): # y_test 와 y_predict 비교하기 위한 함수 (원래의 값과 예측값을 비교)
    return np.sqrt(mean_squared_error(y_test, y_predict)) # 비교하여 그 차이를 빼준다
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기 (1에 가까울 수록 좋다.)
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)
import numpy as num
# num.savetxt('D:\BEER\DeW\data\0822_hsh.csv', y_predict, delimiter=',',)
num.savetxt('test.txt', y_predict, delimiter='///')
'''
y_predict :  [[2.296528 ]
 [1.1204636]
 [3.4305656]
 ...
 [1.253152 ]
 [1.5918205]
 [1.9344857]]
RMSE :  0.8985213181243193
R2 :  0.41898117858759654
'''