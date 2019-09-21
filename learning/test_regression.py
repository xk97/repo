# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 20:07:07 2017

@author: Xianhui
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
#%%
X, y = make_regression(100, n_features=2, noise=0, random_state=11)


def linear_reg(X, W, y, alpha=0.001):
    m = np.shape(X)[0]
    W = np.ones((np.shape(X)[1] + 1, 1))
    X1 = np.hstack((np.ones((m, 1)), X))
    error, tol = [np.inf], 0.001
    while np.linalg.norm(error) > tol:
        y_hat = np.dot(X1, W)
        error = y_hat - np.reshape(y, (-1, 1))
        W = W - alpha * np.dot(X1.T, error) / m
        print(np.linalg.norm(error))
    return W, y_hat

def pred(X, W):
#    m = np.shape(X)[0]
#    return np.dot(np.hstack((np.ones((m, 1)), X)), W)
    return W[0] + np.dot(X, W[1:])


W, y_hat = linear_reg(X, W, y)
pred(X[0, :], W)

plt.plot(X[:, 0], y, 'bs',  marker='.')
plt.show()
plt.plot(X[:, 0], y_hat, 'g^')
plt.show()
