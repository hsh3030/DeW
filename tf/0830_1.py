import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Embedding, Flatten, Input, merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import Model
from time import time
import datetime
from itertools import combinations
import pickle
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

train_label = train['target']
train_id = train['id']
del train['target'], train['id']

test_id = test['id']
del test['id']
print(train_id)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

encoder = LabelEncoder()
train['bin_3'] = encoder.fit_transform(train['bin_3'])
train['bin_4'] = encoder.fit_transform(train['bin_4'])
train['nom_0'] = encoder.fit_transform(train['nom_0'])
train['nom_1'] = encoder.fit_transform(train['nom_1'])
train['nom_4'] = encoder.fit_transform(train['nom_4'])
train['nom_3'] = encoder.fit_transform(train['nom_3'])
train['nom_2'] = encoder.fit_transform(train['nom_2'])
train['ord_1'] = encoder.fit_transform(train['ord_1'])
train['ord_2'] = encoder.fit_transform(train['ord_2'])
train['ord_3'] = encoder.fit_transform(train['ord_3'])
train['ord_4'] = encoder.fit_transform(train['ord_4'])
train['ord_5'] = encoder.fit_transform(train['ord_5'])
train['nom_5'] = encoder.fit_transform(train['nom_5'])
train['nom_6'] = encoder.fit_transform(train['nom_6'])
train['nom_7'] = encoder.fit_transform(train['nom_7'])
train['nom_8'] = encoder.fit_transform(train['nom_8'])
train['nom_9'] = encoder.fit_transform(train['nom_9'])
print(train.head())
test['bin_3'] = encoder.fit_transform(test['bin_3'])
test['bin_4'] = encoder.fit_transform(test['bin_4'])
test['nom_0'] = encoder.fit_transform(test['nom_0'])
test['nom_1'] = encoder.fit_transform(test['nom_1'])
test['nom_4'] = encoder.fit_transform(test['nom_4'])
test['nom_3'] = encoder.fit_transform(test['nom_3'])
test['nom_2'] = encoder.fit_transform(test['nom_2'])
test['ord_1'] = encoder.fit_transform(test['ord_1'])
test['ord_2'] = encoder.fit_transform(test['ord_2'])
test['ord_3'] = encoder.fit_transform(test['ord_3'])
test['ord_4'] = encoder.fit_transform(test['ord_4'])
test['ord_5'] = encoder.fit_transform(test['ord_5'])
test['nom_5'] = encoder.fit_transform(test['nom_5'])
test['nom_6'] = encoder.fit_transform(test['nom_6'])
test['nom_7'] = encoder.fit_transform(test['nom_7'])
test['nom_8'] = encoder.fit_transform(test['nom_8'])
test['nom_9'] = encoder.fit_transform(test['nom_9'])
print(test.head())

print(train.shape)
print(test.shape)
x = np.asarray(train)
y = np.asarray(test)

x = np.reshape()
# from sklearn.model_selection import train_test_split  

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.8, test_size=0.2, shuffle = True)