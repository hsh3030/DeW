# x,y train x = 1~100, y = 501~600
# x,y test x = 1001~1100, y = 1101~1200
import numpy as np    

x_train = np.array([])
for i in range(1, 101):
    x_train = np.append(x_train, i)
print(x_train)

y_train = np.array([])
for i in range(501, 601):
    y_train = np.append(y_train, i)
print(y_train)

x_test = np.array([])
for i in range(1001, 1101):
    x_test = np.append(x_test, i)
print(x_test)

y_test = np.array([])
for i in range(1101, 1201):
    y_test = np.append(y_test, i)
print(y_test)


from keras.models import Sequential
from keras.layers import Dense  
model = Sequential() 

model.add(Dense(44, input_dim = 1, activation = 'relu')) 
model.add(Dense(33)) 
model.add(Dense(22))   
model.add(Dense(11))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 100, batch_size=1) 
loss, acc = model.evaluate(x_test, y_test, batch_size=1) 
print("acc : ", acc)
y_predict = model.predict(x_test) 
print(y_predict)


