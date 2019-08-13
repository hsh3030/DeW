import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# 데이터 읽어 들이기
wine = pd.read_csv("./data/winequality-white.csv", sep=";", encoding="utf-8")

# 데이터를 레이블과 데이터로 분리하기
y = wine["quality"]
x = wine.drop("quality", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 학습하기
'''
n_estimators : 생성할 tree의 개수와

max_features : 최대 선택할 특성의 수입니다.
'''
# model = RandomForestClassifier(n_estimators=600, max_leaf_nodes=1500, n_jobs=-1)
model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=55)
model.fit(x_train, y_train)
aaa = model.score(x_test, y_test)

# 평가하기
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("정답률= ", accuracy_score(y_test, y_pred))
print(aaa)

'''
model= RF() => model = Sequential, models

model.fit(x_train, y_train) => model.fit(x_train, y_train)

model.score(x_test, y_test) => model.evaluate(x_test, y_test)

model. predit(_) => model.predict(새로운x값, 새로운y값)
'''