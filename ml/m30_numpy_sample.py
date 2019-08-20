
####################
import numpy as np
import pandas as pd
'''
a = np.arange(10)
print(a)
np.save("aaa.npy", a)
b = np.load("aaa.npy")
print(b)

################## pima-indians-diabetes ########################
dataset = np.loadtxt("./data/pima-indians-diabetes.csv", delimiter = ",")
np.save("pima-indians-diabetes.npy", dataset)
pima = np.load("pima-indians-diabetes.npy")
print(pima)
#################################################################

######################### iris ##################################
iris_data = pd.read_csv("./data/iris.csv", encoding = 'utf-8')
np.save("iris.npy", iris_data)
iris = np.load("iris.npy")
print(iris)
#################################################################

###################### winequality-white ##########################################
wine = pd.read_csv("./data/winequality-white.csv", sep=",", encoding = "utf-8")
np.save("winequality-white", wine)
wine = np.load("winequality-white.npy")
print(wine)
###################################################################################

######################## mnist ######################################
from keras.datasets import mnist
(X_train, Y_train),(X_test, Y_test) = mnist.load_data()
np.save("mnist.npy", [X_train, Y_train, X_test, Y_test])
mnist = np.load("mnist.npy")
print(mnist)
#####################################################################

########################## cifar10 #################################
from keras.datasets import cifar10
(X_train, Y_train),(X_test, Y_test) = cifar10.load_data()
np.save("cifar10.npy", [X_train, Y_train, X_test, Y_test])
cifar10 = np.load("cifar10.npy")
print(cifar10)
####################################################################

######################## boston_housing ############################
from keras.datasets import boston_housing
(X_train, Y_train),(X_test, Y_test) = boston_housing.load_data()
np.save("boston_housing.npy", [X_train, Y_train, X_test, Y_test])
boston_housing = np.load("boston_housing.npy")
print(boston_housing)
####################################################################

####################### load_boston ################################
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from vecstack import stacking

boston = load_boston()
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=0)

models = [LinearRegression()]
S_train, S_test = stacking(models, X_train, y_train, X_test, 
                           regression=True, n_folds=5,
                           shuffle=False, mode='oof_pred_bag', 
                           random_state=0, verbose=2)


np.save('boston_train.npy', S_train)
np.save('boston_test.npy', S_test)

S_train_loaded = np.load('boston_train.npy')
S_test_loaded = np.load('boston_test.npy')
################################################################3
'''
zoo = pd.read_csv("./DeW/data/data-04-zoo.csv", delimiter = ",")
np.save("zoo.npy", zoo)
iris = np.load("iris.npy")
print(iris)
