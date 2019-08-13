# keras model로 변경하려면 데이터를 np.array로 함수 설
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense  
import numpy as np
# 1. data
x_data = np.array([[0,0], [1,0], [0,1], [1,1]])
y_data = np.array([0,1,1,0])

# 2. model
model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu')) # input_dim(차원) : 1 (input layer) [hidden layer]
model.add(Dense(1, activation='sigmoid'))

# 3. 실행
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_data, y_data, epochs = 2000, batch_size=20)

# 4. 평가 예측

x_test = np.array([[0,0], [1,0], [0,1], [1,1]])
y_test = np.array([0,1,1,0])
y_predict = model.predict(x_test)

loss, acc = model.evaluate(x_test, y_test) # evaluate : 평가 [x,y 값으로]
print("acc : ", acc)
print("loss: ", loss)
# predict : 예측치 확인
# y_predict = model.predict(x_test) 
y_predict = model.predict_classes(x_test) # classes = 0,1로 변경
  
print(y_predict)
