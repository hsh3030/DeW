#1. 데이터 구성
import numpy as np 

x = np.array(range(1, 101))
y = np.array(range(1, 101))
# print(x)

# train 1~60 , val 61~80, val_test 81~100 자르기
# x_train = x[:60]
# x_val = x[60:80]
# x_test = x[80:]
# y_train = y[:60]
# y_val = y[60:80]
# y_test = y[80:]

from sklearn.model_selection import train_test_split  # 함수를 test_size=0.4로 train =0.6 , test = 0.4 로 나눈다
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.4
)
# 함수를 test_size=0.4로 train =0.6 , test = 0.2 val = 0.2 로 나눈다
x_val, x_test, y_val, y_test = train_test_split( 
    x_test, y_test, random_state=66, test_size=0.5
)
print(x_test)
print(y_test)
print(x_val)
print(y_val)

'''
#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense  
model = Sequential() # Sequential = 순차적인 모델

# input_dim : 컬럼의 갯수 (행과는 상관없이 열만 맞으면 적합)
# input_shape=(1, ) - ??행 1열 [데이터 추가 삭제가 용이하다.]
# model.add(Dense(5, input_dim = 1, activation = 'relu'))
model.add(Dense(5, input_shape = (1, ), activation = 'relu')) 
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1)) 

# model.summary() # param = line 갯수 (bias가 하나의 노드)

#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) #loss : 손실율 / optimizer : 적용함수 
model.compile(loss='mse', optimizer='adam', metrics=['mse']) #loss : 손실율 / optimizer : 적용함수 

# model.fit(x, y, epochs = 100, batch_size=3) # epochs : 반복 횟수 / batch_size : 몇개씩 잘라서 할 것인가 / batch_size defalt = 32
model.fit(x_train, y_train, epochs = 100, batch_size=1, validation_data=(x_val, y_val)) # model.fit : 훈련 / validation_data를 추가하면 훈련이 더 잘됨.

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

# R2 구하기 (1에 가까울 수록 좋다.)
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)
'''
