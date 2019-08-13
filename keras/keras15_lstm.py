import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
a = np.array(range(1,11))

# 연결된 함수를 잘라서 사용하는 for문
size = 5 #total size 5개
def split_5(seq, size):
    aaa = []
    for i in range(len(a) - size + 1): # 10-6 을 5개씩 잘라서 ex)[1,2,3,4][5] ,[2,3,4,5][6] 만든다/ i = [0:5] 로 나뉜다
        subset = a[i : (i + size)] # a = [1:6] => [1,2,3,4,5] 출력... [6,7,8,9,10] 출력
        aaa.append([item for item in subset]) # aaa 리스트에 append 해준다 , item = 표현식
    print(type(aaa))
    return np.array(aaa)

dataset = split_5(a, size)
x_train = dataset[:, 0:4] # 6행 4열
y_train = dataset[:, 4 ] # (6, ) 6행
print("==================================")
print(dataset)
print(x_train.shape)  # (6,4)
print(y_train.shape)  # (6, ) 

# reshape [마지막 몇개씩 나눠서 작업할 것인지 정한다]
# x_train = np.reshape(x_train, (6,4,1))
x_train = np.reshape(x_train, (len(a)-size+1, 4, 1))

print(x_train.shape)

x_test = np.array([[[11],[12],[13],[14]], [[12,[13],[14],15]], [[13],[14],[15],[16]], [[14],[15],[16],[17]]])
y_test = np.array([15, 16, 17, 18])

print(x_test.shape)
print(y_test.shape)

# 2. model 구성
model = Sequential()
# return_sequences = True [LSTM을 정의] 연결하여 쓴다.
model.add(LSTM(32, input_shape=(4,1), return_sequences = True))
model.add(LSTM(10, return_sequences = True))
model.add(LSTM(10, return_sequences = True))
model.add(LSTM(10, return_sequences = True))
model.add(LSTM(10, return_sequences = True))
model.add(LSTM(10, return_sequences = True))
model.add(LSTM(10, return_sequences = True))
model.add(LSTM(10, return_sequences = True))
model.add(LSTM(10, return_sequences = True))
model.add(LSTM(10, return_sequences = True))
model.add(LSTM(10, return_sequences = True))
model.add(LSTM(10))

#LSTM -> Dense 층 엮기
model.add(Dense(5, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(1))

model.summary()

# 3. 훈련

model.compile(loss = 'mse', optimizer= 'adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 30, mode = 'auto')
model.fit(x_train, y_train, epochs = 100, batch_size = 1, verbose = 1, callbacks = [early_stopping])
# 4. 평가
loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

print('loss : ', loss)
print('acc : ', acc)
print('y_predict(x_test) : \n', y_predict)



