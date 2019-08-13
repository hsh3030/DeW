#1. 데이터 구성
import numpy as np 

x_train = np.array([1,2,3,4,5,6,7,8,9,10]) # data
y_train = np.array([1,2,3,4,5,6,7,8,9,10]) 
x_test = np.array([11,12,13,14,15,16,17,18,19,20]) # test data
y_test = np.array([11,12,13,14,15,16,17,18,19,20]) # 같은 값으로 검증하지 않기 위해 train 과 test 로 나눈다

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense  
model = Sequential() # Sequential = 순차적인 모델

model.add(Dense(10, input_dim = 1, activation = 'relu')) 
model.add(Dense(4)) 
model.add(Dense(6)) 
model.add(Dense(2))  
model.add(Dense(1))  

# model.summary() # param = line 갯수 (bias가 하나의 노드)

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) #loss : 손실율 / optimizer : 적용함수
# model.fit(x, y, epochs = 100, batch_size=3) # epochs : 반복 횟수 / batch_size : 몇개씩 잘라서 할 것인가 / batch_size defalt = 32
model.fit(x_train, y_train, epochs = 100, batch_size=2) # model.fit : 훈련

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1) # evaluate : 평가 [x,y 값으로]
print("acc : ", acc)

y_predict = model.predict(x_test) # predict : 예측치 확인
print(y_predict)

model.summary()