# -*- coding: utf-8 -*-
# Wine Quality

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import data
dataset = pd.read_csv("./data/winequality-white.csv", sep=";", encoding="utf-8")

# Check for missing values
dataset.isnull().sum()

# value counts of wine quality
dataset.quality.value_counts()

# X and y
X = dataset.iloc[:,0:11].values
y = dataset.iloc[:,11].values

# y is of int type. Change it to categorical
y = y.astype('object')

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
#  one hot encoding
from keras.utils import np_utils
y = np_utils.to_categorical(y)
#y = y[:,1:]

# Splitting the dataset 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Import Keras libraries 
import keras
from keras.models import Sequential
from keras.layers import Dense

# ANN
model = Sequential()

model.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
model.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 10, epochs = 500) # Lesser no of epochs - Basic Model

# Prediction
y_pred = model.predict(X_test)

maxi = y_pred.max(axis=1)
for i in range(len(y_pred)):
    for j in range(7):
        if y_pred[i,j] == maxi[i]:
           y_pred[i,j] = 1
        else:
               y_pred[i,j] = 0
     
# Accuracy    
crt_values = (y_pred == y_test).sum()
wrong_values = (y_pred != y_test).sum()
total = crt_values+wrong_values
result = crt_values/total
print(result) # 86% accuracy