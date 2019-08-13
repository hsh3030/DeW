#1. 데이터 구성
import numpy as np 

x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense 
model = Sequential()

model.add(Dense(100, input_dim = 1, activation = 'relu')) #input_dim(차원) : 1 (input layer)
model.add(Dense(5)) # out-put : 100 -> 5 // layer 추가(node 갯수)(hidden layer)
model.add(Dense(3)) #hidden layer
model.add(Dense(4)) #input = 4 // output = 1 (hidden layer)
model.add(Dense(1)) # output layer

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs = 100, batch_size=1) #model.fit = 모델을 훈련시킨다. x,y를 넣어 / epochs = 훈련횟수 / batch_size = 몇개씩 잘라서 작업 할 것인가?

#4. 평가 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print("acc : ", acc)