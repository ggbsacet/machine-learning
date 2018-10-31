# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 19:42:39 2018

@author: goyal.g
"""

import pandas as pd

data = pd.read_csv('drug200.csv')
data.head()

#get unique value from all columns
print(data['Sex'].unique())

from sklearn import preprocessing
labelsex = preprocessing.LabelEncoder()
labelsex.fit(data['Sex'].unique())
data['Sex']=labelsex.transform(data['Sex'])

labelBP = preprocessing.LabelEncoder()
labelBP.fit(data['BP'].unique())
data['BP'] = labelBP.transform(data['BP'])

labelCholestrol = preprocessing.LabelEncoder()
labelCholestrol.fit(data['Cholesterol'].unique())
data['Cholesterol'] = labelCholestrol.transform(data['Cholesterol'])

X=data.iloc[:,0:5]
Y=data.iloc[:, -1]

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.35, random_state = 3)

xtrain.shape
ytest.shape
ytrain.shape
ytest.shape

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy", max_depth = 5)
dt.fit(xtrain, ytrain)

pt = dt.predict(xtest)
#pt.tolist()


from sklearn import metrics
#import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(ytest, pt))



#Visualize the Tree

##Uncomment the following two lines to lines to install pydotplus
##!conda install -c conda-forge pydotplus -y
##!conda install -c conda-forge python-graphviz -y
#
#from sklearn import metrics
#print("accuracy", metrics.accuracy_score(ytest, pt))
#
#from sklearn.externals.six import StringIO
#import pydotplus
#import matplotlib.image as mpimg
#from sklearn import tree
#%matplotlib inline 
#
#dot_data = StringIO()
#filename = "drugtree.png"
#featureNames = my_data.columns[0:5]
#targetNames = my_data["Drug"].unique().tolist()
#out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png(filename)
#img = mpimg.imread(filename)
#plt.figure(figsize=(100, 200))
#plt.imshow(img,interpolation='nearest')