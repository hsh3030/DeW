# 시각화 작업

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping

IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS =32

BATCH_SIZE = 128
NB_EPOCH = 20
NB_CLASSES = 10
VERBOSE = 1
# validation_split = x_train, y_train 에서 20%를 가져와 validation 한다.
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()
# 데이터 셋 변환 
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

digit = X_train[4825] # mnist image 주소 지정
plt.imshow(digit, cmap = plt.cm.binary)
plt.show()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

X_train1 = X_train.shape[0]
X_test1 = X_test.shape[0]

X_train = X_train.reshape(X_train1, IMG_COLS * IMG_ROWS*IMG_CHANNELS)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_train = X_train.reshape(X_train1, IMG_COLS, IMG_ROWS, IMG_CHANNELS)

X_test = X_test.reshape(X_test1, IMG_COLS * IMG_ROWS*IMG_CHANNELS)

scaler = MinMaxScaler()
scaler.fit(X_test)
X_test = scaler.transform(X_test)

X_test = X_test.reshape(X_test1, IMG_COLS, IMG_ROWS, IMG_CHANNELS)

print(X_train.shape)
print(X_test.shape)


# 기계가 빨리 번역하게 분류한다. (범주형)
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)



# 실수형으로 지정하고 정규화
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255


# 신경망 정의
model = Sequential()
model.add(Conv2D(60, (3,3), padding='same', input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Flatten()) # 1차원으로 펼쳐 dnn으로 변경된다.
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Dense(NB_CLASSES)) #출력부 (출력 output)
model.add(Activation('softmax'))

model.summary()

#학습
model.compile(loss = 'categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='acc', patience=100, mode='auto')
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT, verbose=VERBOSE, callbacks=[early_stopping])

print('Testing....')
score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
print("\nTest score: ", score[0]) # loss
print("Test accuracy: ", score[1]) # acc

#모델저장
print(history.history.keys())
# 단순 정확도에 대한 히스토리 요약
plt.plot(history.history['acc']) 
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
