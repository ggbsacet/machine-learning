# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 14:14:01 2018

@author: goyal.g
"""

"""Pseudo Code

1. Load the data
2. Initialise the value of k
3. For getting the predicted class, iterate from 1 to total number of training data points
    3.1 Calculate the distance between test data and each row of training data. Here we will use Euclidean distance as our distance metric since it’s the most popular method. The other metrics that can be used are Chebyshev, cosine, etc.
    3.2 Sort the calculated distances in ascending order based on distance values
    3.3 Get top k rows from the sorted array
    3.4 Get the most frequent class of these rows
    3.5 Return the predicted class

"""

import pandas as pd
import numpy as np
import math
import operator

#Loading the data
#data = pd.read_csv('telecom-customers.csv')
data = pd.read_csv('iris.csv')



#defiing the function to calculate eucledean distance
def euclideanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)


def knn(trainingSet, testInstance, k): 
    distances = {}
    sort = {}
    length = testInstance.shape[1]
    
    """#### Start of STEP 3 - # Calculating euclidean distance between each row of training data and test data """
    for x in range(len(trainingSet)):        
        dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)
        distances[x] = dist[0]
    
    
    """#### Start of STEP 3.2 - # Sorting them on the basis of distance """
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
        
    neighbors = []
    
    """#### Start of STEP 3.3 - # Extracting top k neighbors """
    for x in range(k):
        neighbors.append(sorted_d[x][0])
    #### End of STEP 3.3
    
    classVotes = {}
    
    """#### Start of STEP 3.4 - # Calculating the most freq class in the neighbors """
    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]
 
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    #### End of STEP 3.4

    """#### Start of STEP 3.5"""
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return(sortedVotes[0][0], neighbors)
    #### End of STEP 3.5
    
testSet = [[7.2, 3.6, 5.1, 2.5]]
test = pd.DataFrame(testSet)

#calculating KNN
k = 1
result = knn(data, test, k)

k = 3
result = knn(data, test, k)

