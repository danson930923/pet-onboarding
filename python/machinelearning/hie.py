import numpy as np 
import pandas as pd
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.preprocessing import MinMaxScaler
import scipy
import pylab
import scipy.cluster.hierarchy

# X1, y1 = make_blobs(n_samples=50, centers=[[4,4], [-2, -1], [1, 1], [10,4]], cluster_std=0.9)

# #Hie
# # agglom = AgglomerativeClustering(n_clusters = 4, linkage = 'average')
# # agglom.fit(X1,y1)
# # Create a figure of size 6 inches by 4 inches.
# # plt.figure(figsize=(6,4))

# # These two lines of code are used to scale the data points down,
# # Or else the data points will be scattered very far apart.

# # Create a minimum and maximum range of X1.
# x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)

# # Get the average distance for X1.
# X1 = (X1 - x_min) / (x_max - x_min)

# dist_matrix = distance_matrix(X1,X1) 
# Z = hierarchy.linkage(dist_matrix, 'average')
# dendro = hierarchy.dendrogram(Z)

##hierarchy.linkage(dist_matrix, 'average') redundant matrix

pdf = pd.read_csv('hieCAR.csv')

# Preprocessing
pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)

featureset = pdf[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]
x = featureset.values
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)
leng = feature_mtx.shape[0]
D = scipy.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j]) #euclidean distance
Z = hierarchy.linkage(D, 'complete')