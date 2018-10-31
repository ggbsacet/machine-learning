# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 12:29:38 2018

@author: goyal.g
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#load the data file
excel = pd.read_csv('FuelConsumptionCo2.csv')

#getting the only required data for MLR
#CO2 Emmision is dependent on multiple variables like.. Engine Size, Cylinders, FuelCombustion
data = excel[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]


#create train and test data by splitting 80-20 method
mask=np.random.rand(len(excel)) < 0.8
train_data=data[mask]
test_data=data[~mask]

from sklearn import linear_model
r = linear_model.LinearRegression()
trainX = np.asanyarray(train_data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
trainY=np.asanyarray(train_data[['CO2EMISSIONS']])
r.fit(trainX, trainY)

print ('Coefficients: ', regr.coef_)
