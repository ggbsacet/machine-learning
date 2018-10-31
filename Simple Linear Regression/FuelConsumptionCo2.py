# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 17:21:10 2018

@author: goyal.g
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

excel = pd.read_csv('FuelConsumptionCo2.csv')

excel.head()

excel.describe()

#slicing data
data1 = excel[['ENGINESIZE','CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

#showing only first 9 rows
#data1.head(9)

data1.hist()
plt.show()

plt.scatter(data1.FUELCONSUMPTION_COMB, data1.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(data1.ENGINESIZE, data1.CO2EMISSIONS, color='red')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()


#craete train and test data to create model
mask=np.random.rand(len(excel)) < 0.8
train_data=data1[mask]
test_data=data1[~mask]

#lets create model

#lets plt map on training data
plt.scatter(train_data.ENGINESIZE, train_data.CO2EMISSIONS)
plt.xlabel("ENGINE SIZE")
plt.ylabel("CO2 Emission")
plt.show()


from sklearn import linear_model
regr = linear_model.LinearRegression()
trainx = np.asanyarray(train_data[["ENGINESIZE"]])
trainy = np.asanyarray(train_data[["CO2EMISSIONS"]])
regr.fit(trainx, trainy)

b0 = regr.intercept_
b1 = regr.coef_

#following will plot the fit line
plt.scatter(train_data.ENGINESIZE, train_data.CO2EMISSIONS, color = "blue")
plt.plot(trainx, trainx*b1[0][0] + b0[0], '-r')
plt.xlabel("ENGINE SIZE")
plt.ylabel("CO2 Emission")
plt.show()


""" Evaluation
we compare the actual values and predicted values to calculate the accuracy of a regression model. Evaluation metrics provide a key role in the development of a model, as it provides insight to areas that require improvement.

There are different model evaluation metrics, lets use MSE here to calculate the accuracy of our model based on the test set: 
    - Mean absolute error: It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand since it’s just average error.
    - Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the squared error. It’s more popular than Mean absolute error because the focus is geared more towards large errors. This is due to the squared term exponentially increasing larger errors in comparison to smaller ones.
    - Root Mean Squared Error (RMSE).
    - R-squared is not error, but is a popular metric for accuracy of your model. It represents how close the data are to the fitted regression line. The higher the R-squared, the better the model fits your data. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
"""
    
#following will work for MSE (Mean Absolute Error)
from sklearn.metrics import r2_score

test_x = np.asanyarray(test_data[['ENGINESIZE']])
test_y = np.asanyarray(test_data[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )


