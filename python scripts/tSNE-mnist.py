import pandas as pd

basepath = "C:\\Gaurav Work\\ML\\machine-learning\\all_datasets_collection\\"
filename = "mnist-train.csv"
data = pd.read_csv(basepath + filename)

data.shape

#seperating feature and labels from data
features = data.drop('label',1)
labels = data['label']

features.shape
labels.shape

from sklearn.manifold import TSNE

#Default perplexity = 30, steps = 1000
model = TSNE(n_components=2, random_state=0)
tsne_data = model.fit_transform(features)

import numpy as np
tsne_data_labels = np.vstack((tsne_data.T,labels)).T

plot_data = pd.DataFrame(tsne_data_labels, columns=['col_1', 'col_2', 'labels'])

import seaborn as sns
import matplotlib.pyplot as plt

sns.FacetGrid(plot_data, hue='labels', size=15).map(plt.scatter, 'col_1', 'col_2').add_legend()
plt.show()

#Lets change the perplexity and iterations
#this will run for perpliexty =5 and iterations = 500
model_30_500 = TSNE(n_components=2, random_state=0, perplexity=5, n_iter=500)
tsne_data_30_500 = model.fit_transform(features)

