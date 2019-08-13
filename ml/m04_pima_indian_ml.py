from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
import numpy
import tensorflow as tf
import os
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# 1. seed 값 생성
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 2. 데이터 로드

print(os.getcwd())

dataset = numpy.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:, 0:8]
Y = dataset[:, 8]

# 3. model 설정
# model = KNeighborsClassifier(n_neighbors=1)
model = KNeighborsRegressor(n_neighbors=1)

# model = SVC()
# model complie
# loss='binary_crossentropy' => sigmoid일때 쓴다.

# 4. model 실행
model.fit(X, Y)

# 5. 결과 출력
x_test = X
y_test = Y
y_predict = model.predict(x_test)

print(x_test, "의 예측결과 : ", y_predict)

print("acc = ", accuracy_score(y_test, y_predict))
