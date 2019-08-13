from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

import numpy as np

k = 5
num_val_samples = len(train_data) // k # 80
num_epochs = 1
all_scores = []

mean = train_data.mean(axis = 0)
train_data -= mean
std = train_data.std(axis = 0)

test_data -= mean
test_data /= std

from keras import models
from keras import layers

def build_model():
    # 동일한 모델을 여러 번 생성할 것이므로 함수를 만들어 사용합니다
    model = models.Sequential()
    model.add(layers.Dense(64, activation = 'relu', input_shape = (train_data.shape[1], )))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['mae'])
    return model

from sklearn.model_selection import KFold

skf = KFold(n_splits = 5)
for train_index, test_index in skf.split(train_data, train_targets):
    partial_train_data, val_data = train_data[train_index], train_data[test_index]
    partial_train_targets, val_targets = train_targets[train_index], train_targets[test_index]
    model = build_model()
    model.fit(partial_train_data, partial_train_targets, epochs = num_epochs, batch_size = 1, verbose = 0)

    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose = 0)
    all_scores.append(val_mae)

    print(all_scores)
    print(np.mean(all_scores))



'''
k = 5
num_val_samples = len(train_data) // k # 80
num_epochs = 1
all_scores = []
for i in range(k):
    print('처리중인 폴드 #', i)
    # 검증 데이터 준비  k번째 분할
    val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples : (i + 1) * num_val_samples]
    # 훈련 데이터 준비 : 다른 분할 전체
    partial_train_data = np.concatenate([train_data[ : i * num_val_samples], train_data[(i + 1) * num_val_samples : ]], axis = 0)
    partial_train_targets = np.concatenate([train_targets[ : i * num_val_samples], train_targets[( i + 1) * num_val_samples : ]], axis = 0)
    # 케라스 모델 구성 (컴파일 포함)
    model = build_model()
    # 모델 훈련 ( verbose = 0 이므로 훈련 과정이 출력되지 않습니다)
    model.fit(partial_train_data, partial_train_targets, epochs = num_epochs, batch_size = 1, verbose = 0) # verbose = 0
    # 검증 세트로 모델 평가
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose = 0)
    all_scores.append(val_mae)
    print(all_scores)
    print(np.mean(all_scores))
'''