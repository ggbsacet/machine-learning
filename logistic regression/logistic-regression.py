# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 15:00:17 2018

@author: goyal.g
"""

import numpy as np
import pandas as pd

excel = pd.read_csv('Churndata.csv')

#orint or list the columns of data
excel.columns

#choose the featured/independent/required columns/variables
data = excel[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 
              'callcard', 'wireless', 'churn']]

#change the churn column data type to int from float/object/DataFrom Series
#to check the data type 
print(type(data.iloc[:,-1]))

#this will give a warning for "SettingwithCopyWarning"
data.iloc[:,-1] = data.iloc[:,-1].astype('int')

#data.head()

x = np.asarray(data[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(data.iloc[:,-1])

from sklearn import preprocessing
x = preprocessing.StandardScaler().fit(x).transform(x)

from sklearn.model_selection import train_test_split
xtrain, ytrain, xtest, ytest = train_test_split(x, y, test_size=0.2, random_state=4)
xtrain.shape
ytrain.shape


from sklearn.linear_model import LogisticRegression
