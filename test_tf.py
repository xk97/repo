# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:37:58 2019
@author: xk97
"""
#%%
import matplotlib as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from tensorflow import set_random_seed
#%%

def seedy(s):
    np.random.seed(s)
    set_random_seed(s)
    
def build_model():
    model = keras.Sequential()
    model.add(Dense(5))
    model.add(keras.layers.Activation('relu'))
    model.add(Dense(3))
    model.add(keras.layers.Activation('sigmoid'))
    model.add(Dense(1))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
#%%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_blobs
    X, y_true = make_blobs(n_samples= 1000, centers=2, random_state=42)
    plt.scatter(X[:,0], X[:,1], c=y_true)
    y_true = y_true[:, np.newaxis]
    X_train, X_test, y_train, y_test = train_test_split(X, y_true)

    from numpy.random import seed
    seed(10)
    import keras
    from keras.layers import Dense, Activation
#    model = build_model()
    model = keras.Sequential()
    model.add(Dense(6))
    model.add(keras.layers.Activation('relu'))
    model.add(Dense(4))
    model.add(keras.layers.Activation('sigmoid'))
    model.add(Dense(1))
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=200, batch_size=750, 
              validation_data=(X_test, y_test), shuffle=False)
    print(f'model eval:{model.evaluate(X_test, y_test)}')
    
    regressor = LogisticRegression(solver='liblinear')
    regressor.fit(X_train, y_train)
    print(f'regressor {regressor.score(X_test, y_test)}')
    # w_trained, b_trained, costs =  
    
    # fig = plt.figure(figsize=(8,6))
    # plt.plot(np.arange(600), costs)
    # plt.title("Development of cost over training")
    # plt.xlabel("Number of iterations")
    # plt.ylabel("Cost")
    # plt.show()


