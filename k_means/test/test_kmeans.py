
import os
import sys
sys.path.append(os.getcwd())  # 相对路径：相对于文件test_*.py的路径

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from k_means import KMeans

x_axis = 'petal_length'
y_axis = 'petal_width'
unique_labels = np.array(['SETOSA', 'VERSICOLOR', 'VIRGINICA'])


data = pd.read_csv('./k_means/data/iris.csv')
data_train = data[[x_axis, y_axis]].values
labels = data['class'].values


def paint_origin_scatter():
    plt.subplot(1,2,1)
    for u_label in unique_labels:
        x = data_train[:,0][labels==u_label]
        y = data_train[:,1][labels==u_label]
        plt.scatter(x, y, label=u_label)
    plt.legend()
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    plt.subplot(1,2,2)
    x = data_train[:,0]
    y = data_train[:,1]
    plt.scatter(x, y)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)    
    plt.show()

def paint_cluster_scatter(centroids,closest_centroid_ids,draw_centroids=False):
    plt.subplot(1,2,1)
    plt.title("Original Class")
    for u_label in unique_labels:
        x = data_train[:,0][labels==u_label]
        y = data_train[:,1][labels==u_label]
        plt.scatter(x, y, label=u_label)
    plt.legend()
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    plt.subplot(1,2,2)
    plt.title("Cluster")
    for centroid_idx in range(centroids.shape[0]):
        cluster_data = data_train[closest_centroid_ids==centroid_idx]
        x = cluster_data[:,0]
        y = cluster_data[:,1]
        plt.scatter(x, y, label=centroid_idx)
    if draw_centroids:
        x = centroids[:,0]
        y = centroids[:,1]
        plt.scatter(x, y, s=30, c='#000000', marker='+', label="centroid" )
    plt.legend()
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()



def test_predict(km):
    data_test = np.array([
        [2,0.6],
        [2,0.7],
        [2,0.8],
        [6,2],
        [5.5,2.1],
    ])
    closest_centroid_ids = km.predict(data_test)
    for centroid_idx in range(km.centroids.shape[0]):
        cluster_data = data_test[closest_centroid_ids==centroid_idx]
        x = cluster_data[:,0]
        y = cluster_data[:,1]
        plt.scatter(x, y, label=centroid_idx)
    x = centroids[:,0]
    y = centroids[:,1]
    plt.scatter(x, y, s=30, c='#000000', marker='+', label="centroid" )
    plt.legend()
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()


km = KMeans(data_train, 3, max_iter=100, random_state=3)
centroids,closest_centroid_ids = km.train()
print(km.inertia)
paint_cluster_scatter(centroids,closest_centroid_ids, draw_centroids=True)

test_predict(km)