import pandas as pd  
import numpy as np

train = pd.read_csv('D:\\big\\train_activity.csv')
test = pd.read_csv('D:\\big\\test1_activity.csv')
train.head()
test.head()
print('train_activity data shape:',train.shape)
print('test_activity data shape:',test.shape)
print(train.info())
print(test.info())
import matplotlib.pyplot as plt
import seaborn as sns
sns.set
def pie_chart(feature):
    feature_ratio = train[feature].value_counts(sort=False)
    feature_size = feature_ratio.size
    feature_index = feature_ratio.index
    death = train[train['death'] == 1][feature].value_counts()
    revive = train[train['revive']== 0][feature].value_counts()
    
    plt.plot(aspect='auto')
    plt.pie(feature_ratio, labels=feature_index, autopct='%1.1f%%')
    plt.title(feature + '\'s ratio in total')
    plt.show()
    
    for i, index in enumerate(feature_index):
        plt.subplot(1, feature_size + 1, i + 1, aspect = 'equal')
        plt.pie([death[index], revive[index]], labels=['Death', 'Revive'], autopct='%1.1f%%')
    
    plt.show()
    pie_chart('party_exp')
    pie_chart('acc_id')