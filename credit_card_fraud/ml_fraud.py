# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 13:07:36 2017

@author: Xianhui
"""
# unbalanced data, stratified kfolds, roc_auc, precision recall
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, ShuffleSplit, StratifiedKFold
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, precision_score, recall_score
print(os.getcwd())


   
#%%

def test():
    df_raw = pd.read_csv('./Data/fraud_data.csv')
    print(df_raw.info(), '\n', df_raw.Class.value_counts())
    sns.countplot(df_raw.Class)
    X, y = df_raw.values[:, :-1], df_raw.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        stratify=df_raw.Class)
    for clf in (DummyClassifier(), LogisticRegression()):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_score = clf.predict_proba(X_test)[:, 1]
        print('train and test score ', clf.score(X_test, y_test),
                clf.score(X_train, y_train))
        print(classification_report(y_test, y_pred))
        print('precision, recall, F1_score', 
                      precision_score(y_test, y_pred),
                      recall_score(y_test, y_pred), 
                      f1_score(y_test, y_pred))
        roc_auc_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred_score)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label='auc {:.2f}'.format(roc_auc))
        plt.xlabel('fpr')
        plt.ylabel('tpr')
        plt.legend()
        plt.show()
        
        plt.figure()
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_score)
        plt.plot(recall, precision, label=clf.__class__)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.legend()
        plt.show()
        
        # stratified kfolds cross validation        
        scores = []
        kfold = StratifiedKFold(n_splits=3)
        kfold.get_n_splits(X, y)
        for k, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
            clf.fit(X[train_idx], y[train_idx])
            score = (clf.score(X[test_idx], y[test_idx]),
                     f1_score(y[test_idx], clf.predict(X[test_idx])))
            scores.append(score)
        print('kfold cv score, f1_score ', scores)
        scores = np.array(scores)
        print(np.mean(scores, axis=0), np.std(scores, axis=0))



if __name__ == '__main__':
    test()





