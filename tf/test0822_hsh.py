import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import matplotlib.pyplot as plt
import random
# 0           date  kp_0h  kp_3h  kp_6h  kp_9h  kp_12h  kp_15h  kp_18h  kp_21h
# 1     1999-01-01      0      2      1      2       2       1       1       1
# 2     1999-01-02      1      2      2      3       3       2       2       1
test_data = pd.read_csv("./data/test0822_hsh.csv", encoding='utf-8', names=['1', '2', '3', '6', '9', '12', '15', '18', '21'])

print(test_data)
print(test_data.shape) # (5480, 9)
print(type(test_data))
print(test_data.info())


# 레이블로 자른다.
y = test_data.loc[:, '21']
x = test_data.loc[:, ['2', '3', '6', '9', '12', '15', '18']]
print(x.shape) # (5480, 7)
print(y.shape) # (5480,)

from sklearn.model_selection import train_test_split
# 학습 전용과 테스트 전용 분리
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, train_size = 0.8, shuffle = True)
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 40, x_test.shape[1]))
print(x_train.shape) # (4384, 7)
print(x_test.shape) # (1096, 7)
print(y_train.shape) # (4384,)
print(y_test.shape) # (1096,)
print(type(x_train))










































































