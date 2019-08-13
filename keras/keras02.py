#1. 데이터 구성
import numpy as np 

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x2 = np.array([4,5,6])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense  
model = Sequential() # Sequential = 순차적인 모델

model.add(Dense(100, input_dim = 1, activation = 'relu')) #input_dim(차원) : 1 (input layer) [hidden layer]
model.add(Dense(28)) # out-put : 100 -> 5 // layer 추가(node 갯수)(hidden layer)
model.add(Dense(15)) #hidden layer
model.add(Dense(7)) #input = 4 // output = 1 (hidden layer)
model.add(Dense(5)) # output layer
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) #loss : 손실율 / optimizer : 적용함수
# model.fit(x, y, epochs = 100, batch_size=3) # epochs : 반복 횟수 / batch_size : 몇개씩 잘라서 할 것인가 / batch_size defalt = 32
model.fit(x, y, epochs = 100)

#4. 평가 예측
loss, acc = model.evaluate(x, y, batch_size=3) # evaluate : 평가 [x,y 값으로]
print("acc : ", acc)

y_predict = model.predict(x2) # predict : 예측치 확인
print(y_predict)
