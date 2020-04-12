# -*- coding: utf-8 -*-
"""

Coursework 3: Clustering

References: 
    https://scikit-learn.org/stable/modules/clustering.html
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
"""

import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

# #############################################################################
# Load data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.4,
                            random_state=0)

plt.plot(X[:,0],X[:,1],'ro', markersize=1)
plt.show()

#%%
"""
Exercise 1: KMeans
"""

# Test different number of clusters
n_clusters = range(2,16)
silhouette = [-1]
for n in n_clusters:
    kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
    labels = kmeans.labels_
    silhouette.append(metrics.silhouette_score(X, labels))


# Plot Silhouette coefficients
plt.plot(range(1,16), silhouette);
plt.xticks(n_clusters);
plt.xlabel('$k$',fontsize = 18)
plt.ylabel(r'$\bar{s}$', fontsize = 18)

# Compute optimum value for the number of clusters
n_clusters = np.argmax(silhouette)+2

# Train optimum model of KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
labels = kmeans.labels_

# Plot optimum model of KMeans
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

plt.figure(figsize=(8,4))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)

plt.title('Fixed number of KMeans clusters: %d' % n_clusters)
plt.show()

#%%

"""
Exercise 2: DBSCAN
"""

# Test different number of epsilon
epsilon = np.linspace(0.1,1,10)

silhouette_euc = []
silhouette_man = []
for eps in epsilon:
    db = DBSCAN(eps=eps, min_samples=10, metric='euclidean').fit(X)
    labels = db.labels_
    if(len(set(labels)) == 1):
        silhouette_euc.append(-1) 
    else:
        silhouette_euc.append(metrics.silhouette_score(X, labels))
        
    db = DBSCAN(eps=eps, min_samples=10, metric='manhattan').fit(X)
    labels = db.labels_
    if(len(set(labels)) == 1):
        silhouette_man.append(-1) 
    else:
        silhouette_man.append(metrics.silhouette_score(X, labels))

# Plot Silhouette coefficients
p1, = plt.plot(epsilon, silhouette_euc, label = 'Euclidean');
p2, = plt.plot(epsilon, silhouette_man, label = 'Manhattan');
plt.xlabel('$\epsilon$', fontsize = 18)
plt.ylabel(r'$\bar{s}$', fontsize = 18)
plt.legend(handles = [p1,p2])
plt.xticks(epsilon);

# Compute optimum value of epsilon
eps_euc = epsilon[np.argmax(silhouette_euc)]
eps_man = epsilon[np.argmax(silhouette_man)]

# Train optimum model of DBSCAN
db = DBSCAN(eps=eps_man, min_samples=10, metric='manhattan').fit(X)
labels = db.labels_

# Plot different characteristics of the model

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# Plot optimum model of DBSCAN
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

plt.figure(figsize=(8,4))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=3)

plt.title('Estimated number of DBSCAN clusters: %d' % n_clusters_)
plt.show()