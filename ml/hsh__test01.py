from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential, Model
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
from sklearn.pipeline import Pipeline
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier

IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS =32

BATCH_SIZE = 128
NB_EPOCH = 20
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train[:300]
y_train = y_train[:300]

y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train = x_train.flatten()
x_test = x_test.flatten()
x_train = x_train.reshape(x_train.shape[0], 1)
x_test = x_test.reshape(x_test.shape[0], 1)

sca = MinMaxScaler()
sca.fit(x_train)
sca.fit(x_test)
x_train = sca.transform(x_train)
x_test = sca.transform(x_test)

x_train = x_train.flatten()
x_test = x_test.flatten()

x_train = x_train.reshape(-1, 32, 32, 3)
x_test = x_test.reshape(-1, 32, 32, 3)
print(x_train.shape)
print(x_test.shape)

def kong(keep_prob = 0.5, optimizer = 'adam'):
    inputs = Input(shape = (IMG_ROWS, IMG_COLS, IMG_CHANNELS), name = 'input')
    x = Sequential()(inputs)
    x = Conv2D(128, (3, 3), padding = 'same', name = 'hidden1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = Dropout(keep_prob)(x)
    x = Flatten()(x)
    x = Dense(256, name = 'hidden2')(x)
    x = Activation('relu')(x)
    x = Dense(256, name = 'hidden3')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, name = 'hidden4')(x)
    prediction = Activation('softmax', name = 'output')(x)
    model = Model(inputs = inputs, outputs = prediction)
    model.compile(optim data_generator = ImageDataGenerator(featurewiseizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
   _center = True, featurewise_std_normalization = True,
                                        rotation_range = 20, width_shift_range = 1.0, height_shift_range = 1.0, 
                                        horizontal_flip = True, vertical_flip = True)
    model.fit_generator(data_generator.flow(x_train, y_train, batch_size = 10), steps_per_epoch = 1000,
                                            epochs = 100, validation_data = (x_test, y_test), verbose = 1)
    return model

def create_hyperparameters():
    batches = [5, 10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    epochs = [10, 50, 100, 300, 500, 700, 1000]
    return{"batch_size" : batches, "optimizer" : optimizers,
           "keep_prob" : dropout, "epochs" : epochs}

model = KerasClassifier(build_fn = kong, verbose = 1)
print(x_train.shape)
hyperparameters = create_hyperparameters()

search = RandomizedSearchCV(model, hyperparameters, 
                            n_iter = 10, n_jobs = -1, cv = 3, verbose = 1)


search.fit(x_train, y_train)

print(search.best_params_)

score = search.score(x_test, y_test)
print("Score : ", score)
