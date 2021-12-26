# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 16:48:26 2021

@author: Zeyuan Li
"""
import os
import requests
from collections import Counter
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn import datasets, neighbors
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from sklearn.metrics import confusion_matrix



# train, test split
def self_split(X, y, r=0.7):
    '''
    This function randomly split the training and testing set by the rate r,
    and the input and output form is np.array

    '''
    # if not np.array. raise error
    assert(isinstance(X,np.ndarray) and isinstance(y,np.ndarray))
    # train size
    n_train = int(np.floor(len(X)*r))
    # shuffle index
    index = np.arange(0,len(X))
    np.random.shuffle(index)
    # split
    X_train = np.array([X[i] for i in index[:n_train]])
    X_test = np.array([X[i] for i in index[n_train:]])
    y_train = np.array([y[i] for i in index[:n_train]])
    y_test = np.array([y[i] for i in index[n_train:]])
    return X_train, X_test, y_train, y_test

def kfold_split_index(X, y ,kfold=10):
    '''
    This function return the kfold split result, but only returns index.
    
    '''
    # if not np.array. raise error
    assert(isinstance(X,np.ndarray) and isinstance(y,np.ndarray))
    # size of one validation set
    fold_size = np.floor(len(X)/kfold)
    # shuffle index
    index = np.arange(0,len(X))
    np.random.shuffle(index)
    result = []
    # train and validation
    for fold in range(1, kfold+1):
        # if not the last fold
        if fold != kfold:
            val_index = index[int(fold_size*(fold-1)):int(fold_size*(fold))]
        # if the last fold
        else:
            val_index = index[int(fold_size*(fold-1)):]
        train_index = set(index).difference(val_index)
        result.append((train_index,val_index))
    return result

def self_KNN(X_train, y_train, X_test, k=5):
    '''
    This function returns y_test, the label of every point in X_test
    '''
    y_pred = []
    for x in X_test:
        d_array = np.sum((X_train-x)**2,axis=1)
        closest_k_point = np.argsort(d_array)[:k]
        closest_k_point_y = [y_train[i] for i in closest_k_point]
        label = Counter(closest_k_point_y).most_common(1)[0][0]
        y_pred.append(label)
    y_pred = np.array(y_pred)
    return y_pred

def self_accuracy(y_pred, y_true):
    error = y_true - y_pred
    right = Counter(error==0).most_common(1)[0][1]
    acc = right/len(y_true)
    return acc
###############################################################################
#---------------------------    main    -------------------------------------##
    
# download data
if os.path.exists('banana.arff'):
    data = arff.loadarff('banana.arff')
else:
    url = 'https://www.openml.org/data/v1/download/1586217/banana.arff'
    r=requests.get(url)
    with open("banana.arff", "wb") as code:
        code.write(r.content)
    data = arff.loadarff('banana.arff')

# data preparation
df = pd.DataFrame(data[0])
df['Class']=df['Class'].astype(int)
# turn data into numpy array
X = df.to_numpy()[:,:2]
y = df.to_numpy()[:,2]    

# split train test
X_train, X_test, y_train, y_test = self_split(X,y,0.9)
# KNN classifier
y_pred = self_KNN(X_train, y_train, X_test, 5)
# test accuracy
acc = self_accuracy(y_pred, y_test)
acc

# choose k through cross validation
train_val_kfold = kfold_split_index(X_train, y_train ,kfold=10)
acc_dict = {}
for k in [3, 5, 7, 11, 13, 17, 19, 21, 23, 29, 31, 37 ]:
    # CV
    mean_acc = 0
    for fold in train_val_kfold:
        cv_X_train = np.array([X_train[i] for i in fold[0]])
        cv_y_train = np.array([y_train[i] for i in fold[0]])
        cv_X_val = np.array([X_train[i] for i in fold[1]])
        cv_y_val = np.array([y_train[i] for i in fold[1]])
        y_val_pred = self_KNN(cv_X_train, cv_y_train, cv_X_val, k)
        mean_acc += self_accuracy(y_val_pred, cv_y_val)
    mean_acc = mean_acc / 10
    acc_dict[k] = mean_acc
acc_dict       

# use the selected k and whole training set to fit model
y_pred = self_KNN(X_train, y_train, X_test, 31)        
# test accuracy
acc = self_accuracy(y_pred, y_test)
acc        
        
        
        
        
        
        