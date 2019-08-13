from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Dense, LSTM, Dropout, Input
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential, Model

cancer = load_breast_cancer()

# 붗꽃 데이터를 레이블과 입력 데이터로 분리하기
y = cancer.target
x = cancer.data

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, shuffle = True)


print(y_train.shape)
print(y_test.shape)

# 모델의 설정
def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(30, ), name='input')
    x = Dense(8, activation='relu', name='hidden1')(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(8, activation='relu', name='hidden2')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(4, activation='relu', name='hidden3')(x)
    x = Dropout(keep_prob)(x)
    prediction = Dense(2, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, output=prediction)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_hyperparameters():
    return{
    "svm__C": [1, 10, 100, 1000], "svm__kernel":["linear"],
    "svm__C": [1, 10, 100, 1000], "svm__kernel":["rbf"], "svm__gamma":[0.001, 0.0001],
    "svm__C": [1, 10, 100, 1000], "svm__kernel":["sigmoid"], "svm__gamma":[0.001, 0.0001]
}

from keras.wrappers.scikit_learn import KerasClassifier # classifier 분류
model = KerasClassifier(build_fn=build_network, verbose=1) # 사이킥런으로 랩핑을 하다.

hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold, GridSearchCV

# estimator=> model을 가져온다.
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])
pipe.fit(x_train, y_train)

search = RandomizedSearchCV(estimator=pipe, param_distributions=hyperparameters,
                            n_iter=10, n_jobs=10, cv=5, verbose=1)# 작업이 n_iter = 10회 수행, cv = 3겹 교차 검증

search.fit(x_train, y_train)

print(search.best_params_)

print('score: ', search.score(x_test, y_test))
print('predict: ', search.predict(x_test))

'''
{'svm__kernel': 'sigmoid', 'svm__gamma': 0.001, 'svm__C': 1000}
score:  1.0
'''
