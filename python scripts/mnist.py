import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the data set
basepath = "C:\\Gaurav Work\\ML\\machine-learning\\all_datasets_collection\\"
filename = "mnist-train.csv"
data = pd.read_csv(basepath + filename)

#checking the data set
data.shape
#it contains 42000 rows and 785 columns

#lets print the first 5 rows
data.head()
#it has class vairable names as "label" and feature variables as "pixlel0-783"


#lets print some of the images
#to do this we need to drop the label column from dataset
#because we need to create this back to 28x28 pixle image
i=199
features = data.drop('label',axis=1)
labels = data['label']

print("Shape of Features matirx - ", features.shape)
print("Shape of Labels matirx - ", labels.shape)

d = features.iloc[i].as_matrix().reshape(28,28)
plt.imshow(d, cmap='gray')
plt.show()

#this is to check whether it(above line) shows correct result corrospoding to label
print(data.iloc[i,0])


#Lets perform PCA with manual steps for Visualization in 2D
#Steps to perform PCA

#1. Get Matirx X and pefrom Column standardization on features (by removing labels)
#2. Calculate Co-variance matirx
#3. Calculate eigan values and eigna vectors
#4. Get top 2 for 2D or 3 for 3D or d for dD eigan vectors
#5. Calculate projection of Matrix X witb Eigan Vectors in 2D, or 3D
#6. Combine ir Add labels with projected matrix
#6. Plot Matrix on graph

#Lets peform Colmun standardization
from sklearn.preprocessing import StandardScaler
col_std_data = StandardScaler().fit_transform(features)
print("Shape of column Standardized Matirx - ", col_std_data.shape)

#Calculate co-variance matix
co_var_matrix = np.matmul(col_std_data.T, col_std_data)
print("Shape of co-variance matrix - ", co_var_matrix.shape)

#caluclate eigan values and vectors and fetch 2 vectors
from scipy.linalg import eigh
eigan_values, eigan_vector = eigh(co_var_matrix, eigvals=(782,783))
print("Shape of eigan vectors - ", eigan_vector.shape)



#Calculate Projection of New points with Matrix
projected_data = np.matmul(eigan_vector.T, col_std_data.T)      # 2x784 . 784x42000 = 2 x 42000
print("Shape of projected matirx - ", projected_data.shape)

#combine or add labels with projected matirx
data_with_labels = np.vstack((projected_data, labels)).T
print("Shape of final pca dta matrix with labels - ", data_with_labels.shape)

#Provide the name of the columns added thorugh PCA
pca_data = pd.DataFrame(data_with_labels, columns=['first_col','second_col','label'])

pca_data.head()

import seaborn as sns

sns.FacetGrid(pca_data, hue="label", size=10).map(plt.scatter, "first_col", "second_col").add_legend()
plt.show()


#Lets perform PCA on data using scikit learn library
from sklearn import decomposition
pca = decomposition.PCA(n_components=2)

pca_data = pca.fit_transform(features)
print("Shape of pca_data is - ", pca_data.shape)

pca_data_labels = np.vstack((pca_data.T, labels)).T
pca_data_labels.shape

plot_data = pd.DataFrame(pca_data_labels, columns=['2_col','1_col','label'])


import seaborn as sns

sns.FacetGrid(plot_data, hue="label", size=10) \
.map(plt.scatter, "1_col", "2_col") \
.add_legend()
plt.show()
