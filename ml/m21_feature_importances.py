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

tree = DecisionTreeClassifier(max_depth=10, random_state=0)
tree.fit(x_train, y_train)

print("훈련 세트 정확도 : {:.3f}".format(tree.score(x_train, y_train)))
print("테스트 세트 정확도 : {:.3f}".format(tree.score(x_test, y_test)))

print("특성 중요도\n", tree.feature_importances_)

# 시각화
import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_cancer(model):
    n_fearures = cancer.data.shape[1]
    plt.barh(np.arange(n_fearures), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_fearures), cancer.feature_names)
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    plt.ylim(-1, n_fearures)

plot_feature_importances_cancer(tree)
plt.show()
