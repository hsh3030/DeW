###################### XOR 분류 ########################### 
# LinearSVC = 선형회귀 model
# SVC model은 XOR 에 적용된다.
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

# 1. data
x_data = [[0,0], [1,0], [0,1], [1,1]]
y_data = [0,1,1,0]

# 2. model
model = SVC() # LinearSVC()는 선형만 되기때문에 SVC로 적용한다.

# 3. 실행

model.fit(x_data, y_data)

# 4. 평가 예측

x_test = [[0,0], [1,0], [0,1], [1,1]]
y_predict = model.predict(x_test)

print(x_test, "의 예측결과 : ", y_predict)

print("acc = ", accuracy_score([0,1,1,0], y_predict))

