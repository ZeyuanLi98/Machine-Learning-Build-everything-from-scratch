# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 16:55:59 2022

@author: Zeyuan Li
"""

import numpy as np
import pandas as pd
import math
from collections import Counter

def self_naive_Bayes_classifier(X_train, y_train,  X_test=None, alpha=1):
    '''
    This function aims to predict the labels of test dataset using
    Naive Bayes Classifier.

    Parameters
    ----------
    X_train : n * d 2D numpy array
        DESCRIPTION.
    y_train : n * 1 1D numpy array
        DESCRIPTION.
    X_test : n * d 2D numpy array, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    y_pred: n * 1 1D numpy array

    '''
    n, d = X_train.shape
    
    # step1 - calculate log-priors log(pi_k)
    size_dict = dict(sorted(dict(Counter(y_train)).items(), 
                               key=lambda x:x[0],reverse=False))
    log_pi = [np.log(size_dict[key]/n) for key in size_dict.keys()]
    
    # step2 - calculate log_pkj, the log probability of feature j appears once in class k
    #         formula: pkj = (nkj + alpha) / (nk + aplha*d)
    # merge the features and labels and convert them into a dataframe
    train_df=pd.concat([pd.DataFrame(X_train),pd.DataFrame(y_train,columns=['y'])],axis=1)
    log_pkj = dict()
    for key in size_dict.keys():
        nkj = train_df[train_df.y==key].iloc[:,0:d].sum(axis=0)
        nk = train_df[train_df.y==key].iloc[:,0:d].sum(axis=0).sum()
        log_pkj[key] = np.log((nkj + alpha) / (nk + alpha * d))
    
    # return
    if X_test is None:
        return log_pi, log_pkj
    else:
        # step3: estimate probability of class k on test dataset
        m = X_test.shape[0]
        c = len(size_dict.keys())
        log_p = np.zeros((m, c))
        for i in range(m):
            x = X_test[i,:]
            for key in size_dict.keys():
                log_p[i,key] = log_pi[key] + x.dot(log_pkj[key])
        p = np.exp(log_p)
        p = p/(np.sum(p,axis=1)).reshape(m,1)
        y_pred = np.argmax(log_p,axis=1)     
        return y_pred, p

def self_accuracy(y_pred, y_true):
    error = y_true - y_pred
    right = Counter(error==0).most_common(1)[0][1]
    acc = right/len(y_true)
    return acc



###############################################################################
#---------------------------    main    -------------------------------------##
'''
X_train has dimensions ntrain by d, and y_train is an ntrain-vector of 0s and 1s,
denoting a message from the cars group or motorcycles group, respectively. The training
set consists of ntrain = 1192 samples, and the test set has ntest = 792 samples. There
are d = 1000 features, where each feature denotes the number of times a particular word
appeared in a document.
'''
X_train = np.load("hw2p2_train_x.npy")
y_train = np.load("hw2p2_train_y.npy")
X_test  = np.load("hw2p2_test_x.npy")
y_test  = np.load("hw2p2_test_y.npy")

# Naive bayes
y_pred, p = self_naive_Bayes_classifier(X_train, y_train, X_test)
self_accuracy(y_pred, y_test)



























