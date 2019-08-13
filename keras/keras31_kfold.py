# 교차검증 [kfold]

from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import StandardScaler, MinMaxScaler

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)
#데이터 표준화
scaler = StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
print(train_data)
'''
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std
'''
from keras import models
from keras import layers

def build_model():
    #동일한 모델을 여러 번 생성할 것이므로 함수를 만들어 사용합니다.
    model = models.Sequential()
    model.add(layers.Dense(64, activation = 'relu', input_shape = (train_data.shape[1],)))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae']) # mae : 음수값을 없앤다.
    return model

seed = 77
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import KFold, cross_val_score # Regressor 회귀방식
model = KerasRegressor(build_fn=build_model, epochs=10, batch_size=1, verbose=1) 
kfold = KFold(n_splits=5, shuffle=True, random_state=seed) # 잘라서 seed 값을 받아 랜덤으로 섞는다.
results = cross_val_score(model, train_data, train_targets, cv=kfold)# fit = cross_val_score, cv(교차검증)

import numpy as np
print(results)
print(np.mean(results))

'''
k = 5 # 5번 돌리고 5번 자른다
num_val_samples = len(train_data) // k # 몫만 남긴다 (80)
num_epochs = 100
all_scores = []
for i in range(k):
    print('처리중인 폴드 #', i)
    #검증 데이터 준비 : k번째 분할
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples] 
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # 훈련 데이터 준비 :  다른 분할 전체
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]],
        axis = 0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
        train_targets[(i + 1) * num_val_samples:]],
        axis = 0)

    # 케라스 모델 구성 (컴파일 포함)
    model = build_model()
    # 모델 훈련(verbose=0 이므로 훈련 과정이 출력되지 않습니다.)
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0) # verbose=0
    # 검증 세트로 모델 평가 (반환 받는다)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

print(val_data.shape)
print(partial_train_data.shape)
print(all_scores)
print(np.mean(all_scores))
'''