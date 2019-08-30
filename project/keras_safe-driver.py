import pandas as pd
import numpy as np
trn = pd.read_csv('C:\\Users\\bitcamp\\.kaggle\\porto-seguro-safe-driver-prediction\\train.csv', na_values=['-1', '-1.0'])
tst = pd.read_csv('C:\\Users\\bitcamp\\.kaggle\\porto-seguro-safe-driver-prediction\\test.csv', na_values=['-1', '-1.0'])

print(trn.shape, tst.shape) 
# (595212, 59) (892816, 58)
print(trn.head())
#    id  target  ps_ind_01  ps_ind_02_cat  ps_ind_03  ps_ind_04_cat  ps_ind_05_cat  ps_ind_06_bin  ...  ps_calc_13  ps_calc_14  ps_calc_15_bin  ps_calc_16_bin  ps_calc_17_bin  ps_calc_18_bin  ps_calc_19_bin  ps_calc_20_bin
# 0   7       0          2            2.0          5            1.0            0.0              0  ...           5           8               0               1               1               0               0
print(trn.info())

np.unique(trn['target'])

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

binary = ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin']