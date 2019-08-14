from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

from keras import models
from keras import layers
from keras.utils import np_utils

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

X_train = X_train.reshape(X_train.shape[0], 32, 32, 3).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3).astype('float32') / 255
from keras.preprocessing.image import img_to_array, array_to_img
X_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in X_train])
X_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in X_test])


from keras.applications import VGG16
conv_base = VGG16(weights = 'imagenet', include_top = False,
                  input_shape=(48, 48, 3))

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) 

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
# model.summary()

history = model.fit(X_train, Y_train, validation_data=(X_test,Y_test), 
                    epochs= 1, batch_size=200, verbose=1)
# 분류모델일 때는 accuracy로 적용한다.
acc = model.predict(X_test)
print(acc)
