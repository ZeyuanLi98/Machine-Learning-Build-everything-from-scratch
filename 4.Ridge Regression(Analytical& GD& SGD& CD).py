# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 14:09:38 2022

@author: Zeyuan Li
"""
import os
from collections import Counter
import pandas as pd
import numpy as np

def feature_standardization(X_test,X_train):
    '''
    Get the standardization rules(mean, std) from X_train, and apply to X_test
    X_test :  m * d 2D numpy array
    X_train : n * d 2D numpy array
    '''
    X_train = X_train.T
    X_test = X_test.T
    col_mean=np.mean(X_train,axis=1).reshape(X_train.shape[0],1)
    col_std=np.std(X_train,axis=1).reshape(X_train.shape[0],1)
    X_standard=(X_test-col_mean)/col_std
    return X_standard.T


def self_Ridge(X, y, lamda, X_test=None, optimizer='Analytical', 
               alpha=0.001, max_iteration=1000):
    '''
    Parameters
    ----------
    X : n * d 2D numpy array with mean 0 and var 1
        DESCRIPTION.
    y : n * 1 1D numpy array
        DESCRIPTION.
    lamda : number, penelty term hyperparameter
        DESCRIPTION.
    X_test : n * d 2D numpy array, optional
        DESCRIPTION. The default is None.
    optimizer : TYPE, optional
        DESCRIPTION. The default is 'Analytical'.
    alpha : number, learning rate, optional
        DESCRIPTION. The default is 0.01.


    '''
    n, d = X.shape
    
    # sub-function1: GD: gradient of J(w)
    def gradient_J(w):
        z = 2*(X.T.dot(X) + lamda * np.identity(d))
        res = z.dot(w)- 2* (X.T.dot(y).reshape(d,1))
        return res
    
    # sub-function2: SGD: gradient for sample i: J_i(w)
    def gradient_J_i(w, i):
        z = y[i]-w.T.dot(X[i,:])
        res = -2*z*X[i,:].T.reshape(d,1) + 2*lamda/n * w
        return res
    
    # method 1: analytical solution
    if optimizer=='Analytical':
        z = X.T.dot(X) + lamda * np.identity(d)
        w = np.linalg.inv(z).dot(X.T).dot(y.reshape(n,1))
        b = np.mean(y) - w.T.dot(np.mean(X,axis=0))
    
    # method 2: Gradient Descent
    elif optimizer=='Gradient Descent' or optimizer=='GD':
        w_list = []
        w_list.append(np.zeros((d,1)))
        for i in range(max_iteration):
            w_old = w_list[-1]
            w_new = w_old- alpha*gradient_J(w_old)
            w_list.append(w_new)
        # final w
        w = w_list[-1]
        b = np.mean(y) - w.T.dot(np.mean(X,axis=0))
    
    # method 3: Stochastic Gradient Descent
    elif optimizer=='Stochastic Gradient Descent' or optimizer=='SGD':
        w_list = []
        w_list.append(np.zeros((d,1)))
        for t in range(max_iteration):
            random_i = [i for i in range(n)]
            np.random.shuffle(random_i)
            for i in random_i:
                w_old = w_list[-1]
                w_new = w_old- alpha*gradient_J_i(w_old, i)
                w_list.append(w_new)
        # final w
        w = w_list[-1]
        b = np.mean(y) - w.T.dot(np.mean(X,axis=0))
    
    # method 4: Coordinate Descent
    elif optimizer=='Coordinate Descent' or optimizer=='CD':
        b = np.mean(y)
        w_list=[]
        w_list.append(np.zeros((d,1)))
        for cycle in range(max_iteration):
            for i in range(d):
                #calculate w_i
                w_old = w_list[-1]
                a_i = 2*sum(list(map(lambda z:z**2,X[:,i])))
                c_i = 0
                for j in range(n):
                    yj_hat_w_no_i = b + X[j,:].dot(w_list[-1])-X[j,i]*w_old[i]
                    c_i += 2 * X[j,i] * (y[j]-yj_hat_w_no_i)
                w_i=c_i/(a_i+2*lamda)
                #update
                w_new = w_old.copy()
                w_new[i] = w_i
                w_list.append(w_new)
        w = w_list[-1]
        
    # return   
    if X_test is None:
        return w, b
    else:
        # predict testm
        y_pred = X_test.dot(w) + b
        return y_pred
    

###############################################################################
#---------------------------    main    -------------------------------------##

# import data
os.chdir("d:\\spyder_workspace")
np.random.seed(0)
X_train = np.load("housing_train_features.npy").T
X_test = np.load("housing_test_features.npy").T
y_train = np.load("housing_train_labels.npy")
y_test = np.load("housing_test_labels.npy")
feature_names = np.load("housing_feature_names.npy", allow_pickle=True)

# standardize
X_train_s = feature_standardization(X_train,X_train)
X_test_s = feature_standardization(X_test,X_train)

# Ridge
# 1. analytical solution
w, b = self_Ridge(X_train_s, y_train, lamda=100, optimizer='Analytical')
y_pred1 = self_Ridge(X_train_s, y_train, lamda=100, X_test = X_test_s, optimizer='Analytical')

# 2. Gradient descent
w, b = self_Ridge(X_train_s, y_train, lamda=100, optimizer='GD',alpha=0.00001, max_iteration=1000)
y_pred2 = self_Ridge(X_train_s, y_train, lamda=100, X_test = X_test_s, optimizer='GD',alpha=0.00001, max_iteration=1000)

# 3. Stochastic Gradient descent
w, b = self_Ridge(X_train_s, y_train, lamda=100, optimizer='SGD',alpha=0.00001, max_iteration=100)
y_pred3 = self_Ridge(X_train_s, y_train, lamda=100, X_test = X_test_s, optimizer='SGD',alpha=0.00001, max_iteration=100)

# 4. Coodinate Descent
w, b = self_Ridge(X_train_s, y_train, lamda=100, optimizer='CD',alpha=0.0001, max_iteration=100)
y_pred4 = self_Ridge(X_train_s, y_train, lamda=100, X_test = X_test_s, optimizer='CD',alpha=0.00001, max_iteration=100)
