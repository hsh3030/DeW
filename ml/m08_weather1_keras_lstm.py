import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 기온 데이터 읽어 들이기
df = pd.read_csv('./data/tem10y.csv', encoding="utf-8")

# 데이터를 학습 전용과 테스트 전용으로 분리하기

interval = 6

# 과거 6일의 데이터를 기반으로 학습할 데이터 만들기
def make_data(data):
    x = []
    y = []
    temps = list(data["기온"])
    for i in range(len(temps)):
        if i < interval: continue
        y.append(temps[i])
        xa = []
        for p in range(interval):
            d = i + p - interval
            xa.append(temps[d])
        x.append(xa)
    return (x, y)

train_x, train_y = make_data(df)
test_x, test_y = make_data(df)

import numpy as np
print(np.array(train_x).shape)
print(np.array(train_y).shape)
print(np.array(test_x).shape)
print(np.array(test_y).shape)
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

from sklearn.model_selection import train_test_split

# train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.4)
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1],1)
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], 1)

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dense, BatchNormalization, Dropout  

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(6,1), return_sequences = True))
model.add(LSTM(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

# loss :  1.5877569439601287
# acc :  1.5877569439601287
# R2 :  0.9746703261416477
# RMSE :  1.2600622174826062

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=50, mode='auto')
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(train_x, train_y, epochs = 300, batch_size= 32, callbacks=[early_stopping])

loss, acc = model.evaluate(test_x, test_y, batch_size=1)

y_predict = model.predict(test_x)

print('loss : ', loss)
print('acc : ', acc)

# R2 구하기 (1에 가까울 수록 좋다.)
from sklearn.metrics import r2_score
r2_y_predict = r2_score(test_y, y_predict)
print("R2 : ", r2_y_predict)

# RMSE 구하기 (RMSE: 낮을수록 좋다.)
from sklearn.metrics import mean_squared_error
def RMSE(test_y, y_predict): # y_test 와 y_predict 비교하기 위한 함수 (원래의 값과 예측값을 비교)
    return np.sqrt(mean_squared_error(test_y, y_predict)) # 비교하여 그 차이를 빼준다
print("RMSE : ", RMSE(test_y, y_predict))

# 결과를 그래프로 그리기
plt.style.use('classic')
plt.figure(figsize=(10, 6), dpi = 100)
plt.plot(test_y, c='r')
plt.plot(y_predict, c='b')
plt.savefig('tenki-kion-lr.png')
plt.show()
