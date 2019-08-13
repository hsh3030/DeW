from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from keras.layers import Dense, LSTM, Dropout, Input
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model

# 데이터 분류
pd = pd.read_csv("./data/wine.csv")

# 붗꽃 데이터를 레이블과 입력 데이터로 분리하기
y = pd.loc[:, "1"]
x = pd.loc[:,["7.4", "0.7", "0", "1.9", "0.076", "11", "34", "0.997", "3.51", "0.56", "9.4", "5"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, shuffle = True)

# 모델의 설정


def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(12, ), name='input')
    x = Dense(8, activation='relu', name='hidden1')(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(8, activation='relu', name='hidden2')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(8, activation='relu', name='hidden3')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(8, activation='relu', name='hidden4')(x)
    x = Dropout(keep_prob)(x)
    prediction = Dense(2, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, output=prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def create_hyperparameters():
    batches = [1, 10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    epochs = [10, 20, 40, 60, 80, 100]
    return{"model__batch_size" : batches, "model__optimizer" : optimizers, "model__keep_prob" : dropout, "model__epochs" : epochs}

from keras.wrappers.scikit_learn import KerasClassifier # classifier 분류
model = KerasClassifier(build_fn=build_network, verbose=1) # 사이킥런으로 랩핑을 하다.

hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold, GridSearchCV

# estimator=> model을 가져온다.
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
# estimator=> model을 가져온다.
pipe = Pipeline([("scaler", MinMaxScaler()), ('model', model)])
search = RandomizedSearchCV(pipe, hyperparameters, n_iter = 10, n_jobs = 15, cv = 5, verbose = 1)

search.fit(x_train, y_train)

print(search.best_params_)
score = search.score(x_test, y_test)
print("Score : ", score)

