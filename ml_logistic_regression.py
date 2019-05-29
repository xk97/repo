# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:37:58 2019
# https://blog.csdn.net/qq_32942549/article/details/79819592
@author: xk97
"""
import numpy as np

class LogisticRegression:
    
    def __init__(self):
        pass
    
    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))
    
    def train(self, X, y_true, n_iters, learning_rate):
        """
        Trains the logistic regression model on given data X and targets y
        """
        # Step 0: Initialize the parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        costs = []

        for i in range(n_iters):
            # Step 1 and 2: Compute a linear combination of the input features
            # and weights,
            # apply the sigmoid activation function
            y_predict = self.sigmoid(np.dot(X, self.weights) + self.bias)

            # Step 3: Compute the cost over the whole training set.
            cost = (- 1 / n_samples) * np.sum(y_true * np.log(y_predict) \
                    + (1 - y_true) * (np.log(1 - y_predict)))

            # Step 4: Compute the gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predict - y_true))
            db = (1 / n_samples) * np.sum(y_predict - y_true)

            # Step 5: Update the parameters
            self.weights = self.weights - learning_rate * dw
            self.bias = self.bias - learning_rate * db

            costs.append(cost)
            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")

        return self.weights, self.bias, costs

    def predict(self, X):
        """
        Predicts binary labels for a set of examples X.
        """
        y_predict = self.sigmoid(np.dot(X, self.weights) + self.bias)
        y_predict_labels = [1 if elem > 0.5 else 0 for elem in y_predict]

        return np.array(y_predict_labels)[:, np.newaxis]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_blobs
    regressor = LogisticRegression()
    X, y_true = make_blobs(n_samples= 1000, centers=2)
    plt.scatter(X[:,0], X[:,1], c=y_true)
    y_true = y_true[:, np.newaxis]
    X_train, X_test, y_train, y_test = train_test_split(X, y_true)
    
    w_trained, b_trained, costs = regressor.train(X_train, y_train, 
                                                  n_iters=600, 
                                                  learning_rate=0.009)
    
    fig = plt.figure(figsize=(8,6))
    plt.plot(np.arange(600), costs)
    plt.title("Development of cost over training")
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.show()



