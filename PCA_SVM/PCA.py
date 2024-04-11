'''Given a 2-dimensional data set (0,2) (2,4), (4,4), (5,4), (7,6). Use PCA in the sklearn package in Python to find two principal components of the data. 
'''
from sklearn.decomposition import PCA
import numpy as np
data=np.array([[0,2],[2,4],[4,4],[5,4],[7,6]])
pca=PCA(n_components=2)œœœœ
pca.fit(data)
principal_components=pca.components_
component1 = principal_components[0]
component2 = principal_components[1]
print("First Principal Component (Weight Vector):", component1)
print("Second Principal Component (Weight Vector):", component2)