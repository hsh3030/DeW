# 1 ~ 100 까지의 숫자를 이용 6개씩 잘라서 rnn구성
# train, test 분리할것
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

x = np.array(range(1, 101))
y = np.array(range(101, 111))

size = 7 #total size 5개
def split_5(seq, size):
    aaa = []
    for i in range(len(x) - size + 1): # 10-6 을 5개씩 잘라서 ex)[1,2,3,4][5] ,[2,3,4,5][6] 만든다/ i = [0:5] 로 나뉜다
        subset = x[i : (i + size)] # a = [1:6] => [1,2,3,4,5] 출력... [6,7,8,9,10] 출력
        aaa.append([item for item in subset]) # aaa 리스트에 append 해준다 , item = 표현식
    print(type(aaa))
    return np.array(aaa)

dataset = split_5(x, size)
x_train = dataset[:, 0:6]
y_train = dataset[:, 6]
print(dataset)
print(x_train)
print(y_train)

print(x_train.shape) # (94, 6)
print(y_train.shape) # (94,)

size1 = 7 #total size 5개
def split_6(seq, size1):
    aaa = []
    for i in range(len(y) - size + 1): # 10-6 을 5개씩 잘라서 ex)[1,2,3,4][5] ,[2,3,4,5][6] 만든다/ i = [0:5] 로 나뉜다
        subset = y[i : (i + size)] # a = [1:6] => [1,2,3,4,5] 출력... [6,7,8,9,10] 출력
        aaa.append([item for item in subset]) # aaa 리스트에 append 해준다 , item = 표현식
    print(type(aaa))
    return np.array(aaa)

dataset1 = split_6(y, size1)
x_test = dataset1[:, 0:6]
y_test = dataset1[:, 6]
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1])
print(x_test)
print(x_test.shape)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
print(x_train.shape)
print(y_test)

model = Sequential()
# return_sequences = True [LSTM을 정의]
model.add(LSTM(16, input_shape=(6,1), return_sequences = True))
model.add(LSTM(10))
#LSTM -> Dense 층 엮기
model.add(Dense(1))
model.summary()
model.compile(loss = 'mse', optimizer= 'adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 30, mode = 'auto')
model.fit(x_train, y_train, epochs = 2000, batch_size = 4, verbose = 1, callbacks = [early_stopping])
# 4. 평가
loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

print('loss : ', loss)
print('acc : ', acc)
print('y_predict(x_test) : \n', y_predict)

# RMSE 구하기 (RMSE: 낮을수록 좋다.)
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): # y_test 와 y_predict 비교하기 위한 함수 (원래의 값과 예측값을 비교)
    return np.sqrt(mean_squared_error(y_test, y_predict)) # 비교하여 그 차이를 빼준다
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기 (1에 가까울 수록 좋다.)
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)