from sklearn.datasets import load_boston

boston = load_boston()
# print(boston.data.shape)
# print(boston.keys())
# print(boston.target)
# print(boston.target.shape)

x = boston.data
y = boston.target

# print(type(boston))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.02)

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200)

# model = LinearRegression()
# model = Lasso()
# model = Ridge()

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(score)









'''
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
boston = load_boston()

# print(boston.data.shape)
# print(boston.key())
# print(boston.target)
# print(boston.target.shape)

x = boston.data
y = boston.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 64)
print(type(boston))

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import accuracy_score
# 모델을 완성하시오.

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(LinearRegression(y_test, y_pred))
print("정답률= ", accuracy_score(y_test, y_pred))
'''
