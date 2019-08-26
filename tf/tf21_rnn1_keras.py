import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

# 데이터 구축

idx2char = ['e', 'h', 'i', 'l', 'o']

_data = np.array([['h', 'i', 'h', 'e', 'l', 'l', 'o']],dtype=np.str).reshape(-1,1)
print(_data.shape) #(7,1)
print(_data)
print(_data.dtype)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray().astype('float32')

x_data = _data[:6,] # (6, 5)
y_data = _data[1:,] # (6, 5)

x_data = x_data.reshape(1, 6, 5)   #(1, 6, 5)
y_data = y_data.reshape(1, 6, 5)

print(x_data.shape) # (1, 6, 5)
print(x_data.dtype)
print(y_data.shape) #(6, )

model = Sequential()
model.add(LSTM(30, input_shape=(6, 5), return_sequences = True))
model.add(LSTM(10, return_sequences = True))
model.add(LSTM(5, activation = "softmax", return_sequences = True))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

history = model.fit(x_data, y_data, epochs= 500, batch_size= 1)

print("\n test acc: %.4f"%(model.evaluate(x_data, y_data)[1]))


pre = model.predict(x_data)
print(pre)
y_data = np.argmax(y_data, axis= 2)
pre = np.argmax(pre, axis = 2)
result_str = [idx2char[c] for c in np.squeeze(pre)]
print("\nPrediction str: ", ''.join(result_str))
