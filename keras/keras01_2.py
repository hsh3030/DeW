#1. 데이터 구성
import numpy as np 

x = np.array([1,2,3])
y = np.array([1,2,3])
x2 = np.array([4,5,6])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense 
model = Sequential()

model.add(Dense(100, input_dim = 1, activation = 'relu')) #input_dim(차원) : 1 (input layer)
model.add(Dense(28)) # out-put : 100 -> 5 // layer 추가(node 갯수)(hidden layer)
model.add(Dense(15)) #hidden layer
model.add(Dense(7)) #input = 4 // output = 1 (hidden layer)
model.add(Dense(5)) # output layer
model.add(Dense(1))
# model.add(Dense(33))
# model.add(Dense(9))
# model.add(Dense(7))
# model.add(Dense(8))
# model.add(Dense(7))
# model.add(Dense(12))
# model.add(Dense(3))
# model.add(Dense(1))


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs = 1000, batch_size=1)

#4. 평가 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print("acc : ", acc)

y_predict = model.predict(x2)
print(y_predict)
