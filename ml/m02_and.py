# LinearSVC = 선형회귀 model
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 1. data
x_data = [[0,0], [1,0], [0,1], [1,1]]
y_data = [0,0,0,1]

# 2. model
model = LinearSVC() # svm에서의 최적화 된 값을 준다. LinearSVC()

# 3. 실행
model.fit(x_data, y_data)

# 4. 평가 예측

x_test = [[0,0], [1,0], [0,1], [1,1]]
y_predict = model.predict(x_test)

print(x_test, "의 예측결과 : ", y_predict)

# y_test = accuracy_score([0,0,0,1] 값과 y_predict 비교하여 acc 값 출력 <분류 모델에서만 쓴다>
print("acc = ", accuracy_score([0,0,0,1], y_predict))

