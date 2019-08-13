############################## 상태유지 LSTM ###############################
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization

#1. DATA
a = np.array(range(1,101))
batch_size = 1

history_mse = []
history_val_mse = []

# LSTM에 넣기 위한 와꾸
size = 5 #total size 5개
def split_5(seq, size):
    aaa = []
    for i in range(len(a) - size + 1): # 10-6 을 5개씩 잘라서 ex)[1,2,3,4][5] ,[2,3,4,5][6] 만든다/ i = [0:5] 로 나뉜다
        subset = a[i : (i + size)] # a = [1:6] => [1,2,3,4,5] 출력... [6,7,8,9,10] 출력
        aaa.append([item for item in subset]) # aaa 리스트에 append 해준다 , item = 표현식
    print(type(aaa))
    return np.array(aaa)

dataset = split_5(a, size)
print("========================================")
print(dataset)
print(dataset.shape)

x_train = dataset[:, 0: 4]
y_train = dataset[:, 4]

x_train = np.reshape(x_train, (len(x_train), size-1, 1))

x_test = x_train + 100
y_test = y_train + 100

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_test[0])

# 2. models 구성 [ 상태 유지 LSTM ] batch_input_shape = (1(batch_size),4,1)
# stateful = True 상태유지를 해라. (초기화 시키지 않겠다.)
model = Sequential()
model.add(LSTM(200, batch_input_shape = (batch_size,4,1), stateful = True))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
th_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq = 0, write_graph=True, write_images=True)
# shuffle=False (그 전의 훈련한 상태를 섞지 않고 가져오겠다..)
# model.reset_states() (상태 유지에선 항상 붙는다.<상태값 자체는 변하지 않음.>)
num_epochs = 100


for epoch_idx in range(num_epochs):
    print('epochs : ' + str(epoch_idx))
    history = model.fit(x_train, y_train, epochs=1, batch_size=batch_size, callbacks=[early_stopping, th_hist],  verbose=2, 
              shuffle=False, validation_data=(x_test,y_test))
    model.reset_states() # 상태유지에서는 fit 할 때마다 넣는다(암기)
    history_mse.append(history.history['mean_squared_error'])
    history_val_mse.append(history.history['val_mean_squared_error'])
mse, _ = model.evaluate(x_train, y_train, batch_size=batch_size)
print("mse : ", mse)
model.reset_states()

y_predict = model.predict(x_test, batch_size=batch_size)

print(y_predict[0:5]) # 앞 5개를 출력

# RMSE 구하기 (RMSE: 낮을수록 좋다.)
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): # y_test 와 y_predict 비교하기 위한 함수 (원래의 값과 예측값을 비교)
    return np.sqrt(mean_squared_error(y_test, y_predict)) # 비교하여 그 차이를 빼준다
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기 (1에 가까울 수록 좋다.)
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

import matplotlib.pyplot as plt
# 그래프에 그리드를 주고 레이블을 표시
plt.plot(history_mse)
plt.plot(history_val_mse)
plt.title('model mse')
plt.xlabel('epoch')
plt.ylabel('mse')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()