import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score

array = np.array(range(1, 101))

size = 8

def split_8(seq, size):
    list = []
    for i in range(len(array) - size + 1):
        subset = array[i : (i + size)]
        list.append([item for item in subset])
    print(type(list))
    return np.array(list)

dataset = split_8(array, size)
print("=====================")

x = dataset[:, 0 : 4]
y = dataset[:, 4 : ]

x = np.reshape(x, (len(array) - size + 1, 4, 1))

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 66, test_size = 0.01)

x_predict = np.array([[[111], [112], [113], [114]]])

model = Sequential()

model.add(LSTM(33, input_shape = (4, 1), return_sequences = True))
model.add(LSTM(16))
model.add(LSTM(16))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(4))

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])

early_stopping = EarlyStopping(monitor = 'loss', patience = 30, mode = 'auto')
model.fit(x_train, y_train, epochs = 100, batch_size = 1, verbose = 1, callbacks = [early_stopping])

loss, acc = model.evaluate(x_test, y_test)

# y_predict = model.predict(x_test)
y_predict = model.predict(x_predict)

print('y_predict(x_test) : \n', y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))

r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)