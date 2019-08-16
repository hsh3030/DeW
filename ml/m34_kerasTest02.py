import os
import pandas as pd 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, MaxPooling1D
from keras.layers import LSTM, Flatten
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
import math
from sklearn.metrics import mean_squared_error
import io
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D
from keras.layers import Embedding
df = pd.read_csv('D:\kospi200test.csv')

scaler = MinMaxScaler()
df[['end']] = scaler.fit_transform(df[['end']])
print(df)
#             day     siga     high      row       end      tra  a-money
# 0    2019-07-31  2036.46  2041.16  2010.95  0.051035   589386   1183.1

price = df['end'].values.tolist()
print(price)



# dataset 생성 함수 정의
window_size = 5
x = []
y = []

for i in range(len(price) - window_size):
    x.append([price[i+j] for j in range(window_size)])
    y.append(price[window_size + i])
    
print(x)
print(y)

x = np.asarray(x)
y = np.asarray(y)
print(x.shape)
print(y.shape)
from sklearn.model_selection import train_test_split  

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=88, test_size=0.2)

x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))



print(x_train.shape) #(475, 1, 5)
print(x_test.shape) #(119, 1, 5)

model = Sequential()
model.add(LSTM(128, input_shape = (1, 5)))
# model.add(Conv1D(475,input_shape=(5,), activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))
model.compile(loss='mse', optimizer='adam')
model.summary()


model.compile(loss='mse', optimizer = 'adam', metrics=['accuracy']) 
model.fit(x_train, y_train, batch_size = 1, nb_epoch = 1000, validation_split = 0.05)

y_predict = model.predict(x_test)
print(y_predict)
