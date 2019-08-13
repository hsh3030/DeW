import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout, Flatten
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv('D:\kospi200test.csv')

print(df.head())
#           day     siga     high      row      end     tra  a-money
# 0  2019-07-31  2036.46  2041.16  2010.95  2024.55  589386   1183.1

high = np.array(df['high'])
siga = np.array(df['siga'])
end = np.array(df['end'])
row = np.array(df['row'])
print(high.shape)
print(siga.shape)
print(end.shape)
print(row.shape)

end = np.reshape(end,(-1, 1))
high = np.reshape(high,(-1, 1))
row = np.reshape(row,(-1, 1))
siga = np.reshape(siga,(-1, 1))
print(high.shape)
print(siga.shape)
print(end.shape)
print(row.shape)

scaler = MinMaxScaler()
scaler.fit(end)
high = scaler.transform(high)
row = scaler.transform(row)
print(high[0], row[0], end[0])
train_data = np.concatenate([high, row], axis=1)
train_data = np.concatenate([train_data, end], axis=1)

print(train_data[0])
size = 6
def split_5(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_5(train_data, size)
dataset = np.reshape(dataset, (dataset.shape[0], dataset.shape[1]* dataset.shape[2], 1))
print(dataset.shape)

x_train = dataset[:, 0:dataset.shape[1]-1]
y_train = dataset[:, dataset.shape[1]-1:]

print(x_train.shape)#(594, 17, 1)
print(y_train.shape)#(594, 1, 1)
y_train = y_train.reshape((y_train.shape[0],))
print(y_train.shape)#(594,)

test_days = 5
end_price = dataset[dataset.shape[0] - test_days:]
end_price += 1

print(dataset[-1])
end_price = np.append(end_price, dataset[-1].reshape((1,dataset[-1].shape[0],dataset[-1].shape[1])),
                                 axis=0)
print("end_price",end_price)
print("end_price",end_price.shape)

x_test = end_price[:, 0:end_price.shape[1]-1]
print(x_test.shape)
y_test = end_price[:, end_price.shape[1]-1:]

model = Sequential()

model.add(LSTM(128, input_shape=(x_train.shape[1], 1), return_sequences=True))
model.add(LSTM(32))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))

model.add(Dense(3, activation='relu'))
# 3. 훈련
model.compile(loss='mse', optimizer='adadelta', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=10, batch_size=10, verbose=1, callbacks=[early_stopping])


# 4. 평가
loss, acc = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print('loss: ', loss)
print('acc: ', acc)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(ytest, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict))   # 낮을수록 좋음

# R2 구하기. R2는 결정계수로 1에 가까울수록 좋음
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print('R2 : ', r2_y_predict)

print('y_predict(x_test) : \n', y_predict)
