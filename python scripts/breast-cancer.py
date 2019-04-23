import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
cancer_data = load_breast_cancer()

print(cancer_data.DESCR)

#convert the dataset into DataFrame
cancer_data = pd.DataFrame(
    data=np.c_[cancer_data['data'], cancer_data['target']], 
    columns=cancer_data['feature_names'].tolist() + ['target'])

#print the shape of data
print(cancer_data.shape)

#print the count of different labels 
print(cancer_data['target'].value_counts())

#Seperate Features and Labels from data
target = cancer_data['target']
features_data = cancer_data.drop('target',1)

#apply pca
#import the required class
from sklearn import decomposition

#setup d = 2 components
pca = decomposition.PCA(n_components=2)

#fit the data 
pca_data = pca.fit_transform(features_data)
print(pca_data.shape)

print(target.shape)

#combine the new pca_features and labels
pca_data_labels = np.vstack((pca_data.T,target)).T
print(pca_data_labels.shape)

plot_data = pd.DataFrame(pca_data_labels, 
        columns=['1_col', '2_col', 'target'])

sns.FacetGrid(plot_data, hue='target', size=6) \
.map(plt.scatter, "1_col", "2_col") \
.add_legend()

plt.show()

pca.explained_variance_ratio_

#Plot Explained Variance
ev_pca = decomposition.PCA(n_components=30)
ev_pca_data = ev_pca.fit_transform(features_data)

ev = ev_pca.explained_variance_
per_ev = ev_pca.explained_variance_ / np.sum(ev_pca.explained_variance_)

cum_ev = np.cumsum(per_ev)
plt.plot(cum_ev)




plt.matshow(pca.components_,cmap='viridis')
plt.yticks([0,1],['1_col','2_col'],fontsize=10)
plt.colorbar()
plt.xticks(range(len(cancer_data.columns)),
    cancer_data.columns,rotation=65,ha='left')
plt.tight_layout()
plt.show()#)

