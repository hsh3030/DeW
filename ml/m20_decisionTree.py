from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

# tree = DecisionTreeClassifier(random_state=0)
# tree.fit(x_train, y_train)

# print("훈련 세트 정확도 : {:.3f}".format(tree.score(x_train, y_train)))
# print("테스트 세트 정확도 : {:.3f}".format(tree.score(x_test, y_test)))

tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(x_train, y_train)

print("훈련 세트 정확도 : {:.3f}".format(tree.score(x_train, y_train)))
print("테스트 세트 정확도 : {:.3f}".format(tree.score(x_test, y_test)))

print("특성 중요도\n", tree.feature_importances_)

'''
훈련 세트 정확도 : 1.000테스트 세트 정확도 : 0.937 => 훈련이 과적합 되어 있다.
'''