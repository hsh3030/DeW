########### numpy 저장.npy ########
import numpy as np
a = np.arange(10)
print(a)
np.save("aaa.npy", a)
b = np.load("aaa.npy")
print(b)
##################################

########## model save ###############
model.save('savetest01.h5')
#####################################

########## model load ##############
from keras.models import load_model
model = load_model("savetest01.h5")
from keras.layers import Dense
model.add(Dense(1))
#####################################

############### pands를 numpy로 바꾸기 ##################
판다스.value

######## csv 불러오기 ################
dataset = numpy.loadtxt("./DeW/data/data-04-zoo.csv", delimiter = ",")
iris_data = pd.read_csv("./data/iris.csv", encoding = 'utf-8')
            # index_col = 0, encoding = 'cp949', sep=",", header=None
            # names = ['x1','x2','x3','x4','y']
wine = pd.read_csv("./data/winequality-white.csv", sep=",", encoding = "utf-8")
################################################

############### utf-8 #######################
#-*-coding:utf-8-*-
#########################################

############## 각종 샘플 데이터 셋 #####################
from keras.datasets import mnist
(X_train, Y_train),(X_test, Y_test) = mnist.load_data()

from keras.datasets import cifar10
(X_train, Y_train),(X_test, Y_test) = cifar10.load_data()

from keras.datasets import boston_housing
(X_train, Y_train),(X_test, Y_test) = boston_housing.load_data()

from sklearn.datasets import load_boston
boston = load_boston()
print(boston.keys())   # data, target
boston.data # x값, 넘파이
boston.target # y값, 넘파이

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
##########################################################