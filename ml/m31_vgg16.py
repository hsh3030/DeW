############# 다른 사람 파일 당겨오기 ################
from keras.applications import VGG16
conv_base = VGG16(weights = 'imagenet', include_top = False,
                  input_shape=(150, 150, 3))
# conv_base = VGG16() # 244,244,3


from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras import layers

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation = 'relu', batch_size = (150, 150, 3)))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()


