import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime


data_train = pd.read_csv('./train.csv')
data_test = pd.read_csv('./test.csv')

print(data_train['target'].value_counts())

# 대상 및 ID 삭제 및 카테고리 열 선택
y = data_train['target']
data_id = data_test['id']

data_train=data_train.drop(['id','target'],axis=1)
data_test=data_test.drop(['id'],axis=1)

cate_cols = [cols for cols in data_train.columns if data_train[cols].dtype == 'object']

# 범주 열 인코딩 레이블 
encoder = LabelEncoder()
for col in cate_cols:
    data_train[col] = pd.DataFrame(encoder.fit_transform(data_train[col]))
    data_test[col] = pd.DataFrame(encoder.fit_transform(data_test[col]))   
x_train,x_valid,y_train,y_valid = train_test_split(data_train,y,random_state=1)

scaler = StandardScaler()
data_train = scaler.fit_transform(data_train)
data_test = scaler.fit_transform(data_test)

# 200 개의 추정기와 2 개의 스케일 위치 가중치로 모델 설명 (불균형 데이터의 경우)
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

clf = XGBClassifier(n_estimators=5000,scale_pos_weight=2,random_state=1,colsample_bytree=0.5, eta=0.2, max_depth=8, learning_rate= 0.1, metric = 'auc',tree_method = 'gpu_hist')

X = data_train
y = y
skf = StratifiedKFold(n_splits=5, random_state=1024, shuffle=False)

for train_index, val_index in skf.split(X, y):
  x_train, x_valid = X[train_index], X[val_index]
  y_train, y_valid = y[train_index], y[val_index]
  clf.fit(x_train,y_train)
# clf.fit(x_train,y_train)

predictions = clf.predict_proba(x_valid)[:,1]

# roc_auc_score를 사용하여 점수 계산
score = roc_auc_score(y_valid,predictions)
print(score)

predict = clf.predict_proba(data_test)[:,1]

submission = pd.DataFrame({'id': data_id, 'target': predict})
submission.to_csv('XGB_submission.csv', index=False)