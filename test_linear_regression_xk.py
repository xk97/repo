# -*- coding: utf-8 -*-
"""
Spyder Editor
Date: 11/4/2017
author: Xianhui.
"""

import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

#%%

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_reg(X, y, alpha=0.001):
    m, p = np.shape(X)
    X1 = np.copy(X)
    X1 = np.insert(X, 0, 1, axis=1)
    y1 = y.reshape((-1, 1))
    W = np.ones((p + 1, 1))
    for i in range(2000):
        z = np.dot(X1, W)
        y_hat = sigmoid(z)
        error = y_hat - y1
        W = W - alpha * np.dot(X1.T, error) / m
        if not i % 100:
            print(np.linalg.norm(error))
    return W, y_hat

def pred(X, W):
    return sigmoid(W[0] + np.dot(X, W[1:])) > 0.5

if __name__  == "__main__":
    X, y = make_classification(n_samples=100, 
                               n_features=3,
                               n_redundant=0,
                               random_state=11)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='.', s=200)
    W, y_hat = logistic_reg(X, y, alpha=0.05)
    pred([-2, -2, -2], W)
