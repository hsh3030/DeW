import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore') # 경고무시

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("./data/iris2.csv", encoding="utf-8")

# 붗꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:,["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

# classifier 알고리즘 모두 추출하기 === 1
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter="classifier") # all_estimators => 모든 sklearn에 있는 model이 전부 들어가있다.
# allAlgorithms = all_estimators(type_filter="regressor")

# k-분할 크로스 밸리데이션 전용 객체
kfold_cv = KFold(n_splits=5, shuffle=True) # n_splits=5 => 5조각으로 나눈다. 4조각은 train, 1조각 test 셋

print(allAlgorithms)
print(len(allAlgorithms))
print(type(allAlgorithms))

for(name, algorithm) in allAlgorithms:

    # 각 알고리즘 객체 생성하기 === 2
    clf = algorithm()

    # score 매서드를 가진 클래스를 대상으로 하기
    if hasattr(clf,"score"): # 변수가 있는지 확인[clf에 score라는 멤버가 있는지 확인]

        # 크로스 밸리데이션
        scores = cross_val_score(clf, x, y, cv= kfold_cv)
        print(name,"의 정답률= ")
        print(scores) # scores => classifier = acc, regressor = R2

'''        
4 => MLPClassifier 의 정답률= [1,0.89473684, 1,1] ave = 0.99618421

5 => SVC 의 정답률= [0.9,1,0.96666667, 1,1] ave = 0.97333334‬

6 => LinearDiscriminantAnalysis 의 정답률= [1,0.96, 1,1,0.96,0.96] ave = 0.98

7 =>LinearDiscriminantAnalysis 의 정답률= [1,1,0.95454545, 1, 1, 1,0.9047619] ave = 0.97990105

8 => MLPClassifier 의 정답률= [0.94736842, 0.89473684,1,1,1,1, 1, 1] ave = 0.9802631575‬

9 => MLPClassifier 의 정답률= [0.94117647, 1,1, 1,0.94117647, 1,1, 1,1,] ave = 0.98692810444444444444444444444444‬

10 => SVC 의 정답률= [0.93333333, 0.93333333, 1, 1, 0.93333333, 1, 1, 1,1, 1] ave = 0.97999999‬
'''