#1. 데이터 구성
import numpy as np 

x_train = np.array([1,2,3,4,5,6,7,8,9,10]) # data (10행 1열) ****행무시 열이 우선****
y_train = np.array([1,2,3,4,5,6,7,8,9,10]) 
x_test = np.array([11,12,13,14,15,16,17,18,19,20]) # test data
y_test = np.array([11,12,13,14,15,16,17,18,19,20]) # 같은 값으로 검증하지 않기 위해 train 과 test 로 나눈다
x3 = np.array([101,102,103,104,105,106]) # 6행 1열
x4 = np.array(range(30,50)) #range(30, 50) [50-1] 30~49

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense  
model = Sequential() # Sequential = 순차적인 모델

# input_dim : 컬럼의 갯수 (행과는 상관없이 열만 맞으면 적합)
# input_shape=(1, ) - ??행 1열 [데이터 추가 삭제가 용이하다.]
# model.add(Dense(15, input_dim = 1, activation = 'relu'))
model.add(Dense(10, input_shape = (1, ), activation = 'relu')) 
model.add(Dense(12))
model.add(Dense(11)) 
model.add(Dense(8)) 
model.add(Dense(7)) 
model.add(Dense(4)) 
model.add(Dense(2))  
model.add(Dense(1))  

# model.summary() # param = line 갯수 (bias가 하나의 노드)

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) #loss : 손실율 / optimizer : 적용함수 
# model.fit(x, y, epochs = 100, batch_size=3) # epochs : 반복 횟수 / batch_size : 몇개씩 잘라서 할 것인가 / batch_size defalt = 32
model.fit(x_train, y_train, epochs = 100, batch_size=1) # model.fit : 훈련

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1) # evaluate : 평가 [x,y 값으로]
print("acc : ", acc) # acc = 분류 모델에 적용

y_predict = model.predict(x_test) # predict : 예측치 확인
print(y_predict)

# RMSE 구하기 (RMSE: 낮을수록 좋다.)
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): # y_test 와 y_predict 비교하기 위한 함수 (원래의 값과 예측값을 비교)
    return np.sqrt(mean_squared_error(y_test, y_predict)) # 비교하여 그 차이를 빼준다
print("RMSE : ", RMSE(y_test, y_predict))
