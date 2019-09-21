# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 19:14:35 2017

@author: Xianhui
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

#%% for fun

def plot_cluster(X, y, centers=None):
    labels = np.unique(y)
    for label in labels:
        plt.scatter(X[y == label, 0], X[y == label, 1], label=label)
    plt.legend(labels)
    plt.scatter(centers[:, 0], centers[:, 1], marker='+', s=150)
    plt.show()

def random_center(X, k=3):
    centers = np.zeros((k, np.shape(X)[1]))
    for i in range(np.shape(X)[1]):
        centers[:, i] = np.random.uniform(np.min(X[:, i]),
                                    np.max(X[:, i]),
                                    size=k)
    return centers

def distance_ecu(X, centers):
    np.random.seed(8)
    n = np.shape(X)[0]
    distance = np.zeros((n, len(centers)))
    for i, center in enumerate(centers):
        distance[:, i] = np.linalg.norm(X - np.tile(center, (n, 1)),
                                        axis=1)
    y_label = np.argmin(distance, axis=1)
    return y_label

def kmeans(X, k=3):
    d, tol = [np.inf], 0.1
    centers = random_center(X, k)
    centers_new = np.zeros(np.shape(centers))
    while sum(d) > tol:
        y_label = distance_ecu(X, centers)
        for i in range(len(centers)):
            centers_new[i] = np.mean(X[y_label == i], axis=0)
        d = np.linalg.norm(centers_new - centers, axis=1)
        centers = np.copy(centers_new)
        print(d)
    return y_label, centers

if __name__ == "__main__":
    X, y = make_blobs(n_samples=200, 
                      n_features=2, 
                      centers=3, 
                      random_state=11)
    centers = random_center(X, k=3)
    plot_cluster(X, y, centers)
    y_label, centers = kmeans(X, 3)
    plot_cluster(X, y_label, centers)

