# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 17:30:34 2018

@author: goyal.g
"""

import pandas as pd
import numpy as np

data = pd.read_csv('iris.csv')

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(data.iloc[:,0:4], data['Name'])

testSet = [[7.2, 3.6, 5.1, 2.5]]
test = pd.DataFrame(testSet)

# Predicted class
print(neigh.predict(test))


# 3 nearest neighbors
print(neigh.kneighbors(test)[1])