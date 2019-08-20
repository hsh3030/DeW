from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
from keras.models import Sequential

xy = np.loadtxt('./DeW/data/data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
print(x_data.shape, y_data.shape)

y_data = np_utils.to_categorical(y_data)
print(x_data.shape, y_data.shape)

model = Sequential
model.add(Dense(256, input_shape = (7,), activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) 
model.fit(x_data, y_data, validation_data=(x_data,y_data), 
                    epochs= 1, batch_size=2000, verbose=1)

print("\n Test Accuracy: %.4f" % (model.evaluate(x_data, y_data)[1]))
