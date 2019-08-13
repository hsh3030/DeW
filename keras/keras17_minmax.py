#스케일링¶
# 스케일링은 자료 집합에 적용되는 전처리 과정으로 모든 자료에 선형 변환을 적용하여 전체 자료의 분포를 평균 0, 분산 1이 되도록 만드는 과정이다.
# 스케일링은 자료의 오버플로우(overflow)나 언더플로우(underflow)를 방지하고 독립 변수의 공분산 행렬의 조건수(condition number)를 감소시켜 최적화 과정에서의
# 안정성 및 수렴 속도를 향상시킨다.

# scikit-learn에서는 다음과 같은 스케일링 클래스를 제공한다.

# StandardScaler(X): 평균이 0과 표준편차가 1이 되도록 변환.
# RobustScaler(X): 중앙값(median)이 0, IQR(interquartile range)이 1이 되도록 변환.아웃라이어의 영향을 최소화
# MinMaxScaler(X): 최대값이 각각 1, 최소값이 0이 되도록 변환
# MaxAbsScaler(X): 0을 기준으로 절대값이 가장 큰 수가 1또는 -1이 되도록 변환

# 1. StandardScaler
# 평균을 제거하고 데이터를 단위 분산으로 조정한다. 그러나 이상치가 있다면 평균과 표준편차에 영향을 미쳐 변환된 데이터의 확산은 매우 달라지게 된다.
# 따라서 이상치가 있는 경우 균형 잡힌 척도를 보장할 수 없다.

# from sklearn.preprocessing import StandardScaler
# standardScaler = StandardScaler()
# print(standardScaler.fit(train_data))
# train_data_standardScaled = standardScaler.transform(train_data)

# 2. MinMaxScaler
# 모든 feature 값이 0~1사이에 있도록 데이터를 재조정한다. 다만 이상치가 있는 경우 변환된 값이 매우 좁은 범위로 압축될 수 있다.
# 즉, MinMaxScaler 역시 아웃라이어의 존재에 매우 민감하다.

# from sklearn.preprocessing import MinMaxScaler
# minMaxScaler = MinMaxScaler()
# print(minMaxScaler.fit(train_data))
# train_data_minMaxScaled = minMaxScaler.transform(train_data)

# 3. MaxAbsScaler
# 절대값이 0~1사이에 매핑되도록 한다. 즉 -1~1 사이로 재조정한다. 양수 데이터로만 구성된 특징 데이터셋에서는 MinMaxScaler와 유사하게 동작하며, 큰 이상치에 민감할 수 있다.

# from sklearn.preprocessing import MaxAbsScaler
# maxAbsScaler = MaxAbsScaler()
# print(maxAbsScaler.fit(train_data))
# train_data_maxAbsScaled = maxAbsScaler.transform(train_data)

# 4. RobustScaler
# 아웃라이어의 영향을 최소화한 기법이다. 중앙값(median)과 IQR(interquartile range)을 사용하기 때문에 StandardScaler와 비교해보면 표준화 후 
# 동일한 값을 더 넓게 분포 시키고 있음을 확인 할 수 있다.
# IQR = Q3 - Q1 : 즉, 25퍼센타일과 75퍼센타일의 값들을 다룬다.

# from sklearn.preprocessing import RobustScaler
# robustScaler = RobustScaler()
# print(robustScaler.fit(train_data))
# train_data_robustScaled = robustScaler.transform(train_data)

# RNN 에 LSTM 은 포함된 상태
from numpy import array # as np 대신 바로 array 가져와 쓴다
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터 만들기
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],[20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

from sklearn.preprocessing import StandardScaler, MinMaxScaler
# StandardScaler, MinMaxScaler 사용법
# x 를 transform을 하면 다른 변수는 fit 할 필요 없이 transfrom(ex>x_test) 로 넣어주면 된다.
# scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
print(x)

'''
print("x.shape : ", x.shape)
print("y.shape : ", y.shape)

# reshape 작업
x = x.reshape((x.shape[0], x.shape[1],1)) # x.shape[0] = 4행 , x.shape[1] = 3열 , 1 = 자르는 갯수 // y.shape는 결과값의 갯수로 생각 (4,)
print("x.shape : ", x.shape)

# 2. Model 구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1))) # (3,1) ?행 3열 dim값 = 1
model.add(Dense(4))
model.add(Dense(6))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(3))
model.add(Dense(1))

# model.summary()
# 3. 훈련 실행 (lstm에서는 layer과 node의 수 보다 epoch를 더 할 수록 결과값이 좋을 수 있다.)
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs = 5000, batch_size= 1) # model.fit : 훈련 / validation_data를 추가하면 훈련이 더 잘됨.

x_input = array([25,35,45]) # 1,3, ????
x_input = x_input.reshape((1,3,1)) 

yhat = model.predict(x_input)
print(yhat)
'''
