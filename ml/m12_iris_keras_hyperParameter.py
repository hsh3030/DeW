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
iris_data = pd.read_csv("./data/iris2.csv", encoding="utf-8")

# 붗꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:,["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, shuffle = True)
kfold_cv = KFold(n_splits=5, shuffle=True)

# 모델의 설정

def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(4, ), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(64, activation='relu', name='hidden4')(x)
    x = Dropout(keep_prob)(x)
    prediction = Dense(3, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, output=prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def create_hyperparameters():
    batches = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    optimizers =['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size":batches, "optimizer":optimizers, "keep_prob":dropout}

from keras.wrappers.scikit_learn import KerasClassifier # classifier 분류
model = KerasClassifier(build_fn=build_network, verbose=1) # 사이킥런으로 랩핑을 하다.

hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold, GridSearchCV

# estimator=> model을 가져온다.
search = RandomizedSearchCV(estimator=model, param_distributions=hyperparameters, n_iter=10, n_jobs=1, cv=3, verbose=1)# 작업이 n_iter = 10회 수행, cv = 3겹 교차 검증

search.fit(x_train, y_train)

print(search.best_params_)