import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# 데이터 읽어 들이기
wine = pd.read_csv("./data/winequality-white.csv", sep=";", encoding="utf-8")

# 데이터를 레이블과 데이터로 분리하기
y = wine["quality"]
x = wine.drop("quality", axis=1)

# y 레이블 변경하기
newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
tree = XGBClassifier(booster = 'gbtree', max_depth=20 ,gamma = 0.1, silent = 0, eta = 0.3, random_state = 0)
tree.fit(x_train, y_train)
print("훈련 세트 정확도 : {:.3f}".format(tree.score(x_train, y_train)))
print("테스트 세트 정확도 : {:.3f}".format(tree.score(x_test, y_test)))
print("특성 중요도\n", tree.feature_importances_)
# 학습하기
parameters = {
    "max_depth": [1, 10, 20, 30],  "gamma":[0.001, 0.0001], "booster": ['gbtree'],
    "loss": ['deviance', 'exponential']
}
# model = RandomForestClassifier(n_estimators=500, max_leaf_nodes=None, n_jobs=-1)
kfold_cv = KFold(n_splits=5, shuffle=True)
model = RandomizedSearchCV(XGBClassifier(), parameters, cv=kfold_cv)
model.fit(x_train, y_train)
aaa = model.score(x_test, y_test)

# 평가하기
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("정답률= ", accuracy_score(y_test, y_pred))
print(aaa)
'''
훈련 세트 정확도 : 1.000
테스트 세트 정확도 : 0.944
특성 중요도
 [0.08837243 0.11192286 0.08085214 0.09684006 0.0879816  0.11032311
 0.07310185 0.07630362 0.08399446 0.07418722 0.11612073]
정답률=  0.9295918367346939
0.9295918367346939
'''