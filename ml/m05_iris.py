import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


# 붗꽃 데이터 읽어 들이기
iris_data = pd.read_csv("./data/iris.csv", encoding='utf-8', names=['a', 'b', 'c', 'd', 'y'])
print(iris_data)
print(iris_data.shape)
print(type(iris_data))

# loc는 레이블로 자른다, iloc는 열로 자른다.
y = iris_data.loc[:, "y"]
x = iris_data.loc[:,["a", "b", "c", "d"]]

# y2 = iris_data.iloc[:, 4]
# x2 = iris_data.iloc[:, 0:4]
print(x.shape) # (150, 4)
print(y.shape) # (150,)



# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, train_size = 0.7, shuffle = True)
print(x_train.shape)
print(x_test.shape)

print(y_test) # str 형식으로 저장되어 있음.

# 학습하기
# clf = SVC()
clf = LinearSVC()
clf.fit(x_train, y_train)

# 평가하기
y_pred = clf.predict(x_test)
print("정답률 : ", accuracy_score(y_test, y_pred))
