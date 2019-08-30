import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Embedding, Flatten, Input, merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import Model
from time import time
import datetime
from itertools import combinations
import pickle
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

train_label = train['target']
