# keras model로 변경하려면 데이터를 np.array로 함수 설
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense  
import numpy as np
# 1. data
x_data = np.array([[0,0], [1,0], [0,1], [1,1]])
y_data = np.array([0,0,0,1])

# 2. model
model = Sequential() # svm에서의 최적화 된 값을 준다. LinearSVC()
model.add(Dense(60, input_shape=(2,), activation='sigmoid')) # input_dim(차원) : 1 (input layer) [hidden layer]
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(1))

# 3. 실행
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_data, y_data, epochs = 100, batch_size=20)

# 4. 평가 예측

x_test = np.array([[0,0], [1,0], [0,1], [1,1]])
y_test = np.array([0,0,0,1])
y_predict = model.predict(x_test)

loss, acc = model.evaluate(x_test, y_test) # evaluate : 평가 [x,y 값으로]
print("acc : ", acc)
print("loss: ", loss)

# y_predict = model.predict(x_test)
y_predict = model.predict_classes(x_test) # classes = 0,1로 변경
print(y_predict)

