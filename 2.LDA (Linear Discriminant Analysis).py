# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 14:42:14 2021

@author: Zeyuan Li
"""
import numpy as np
from collections import Counter
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

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

def self_accuracy(y_pred, y_true):
    error = y_true - y_pred
    right = Counter(error==0).most_common(1)[0][1]
    acc = right/len(y_true)
    return acc

def self_LDA(X_train = None, y_train = None, X_test = None, pi=None, mu=None, Sigma=None, n_components = None):
   
    '''
    This function returns the result of LDA (Linear Discriminant Analysis)
    If func = 'classifier', then returns the classification model,
    if func = 'dimension reduction', then returns the training data X after dimension
    reduction
    
    X_train : n * d 2D numpy array.
    y_train : n * 1 1D numpy array

    '''
      
    ############## if true, LDA classifier   #######################
    if n_components is None:
        # if train parameters are not plugged in, then compute parameters
        if (pi is None) and (mu is None) and (Sigma is None): 
            n, d = X_train.shape 
            # order based on labels(0,1,2...) and get a dict
            size_dict = dict(sorted(dict(Counter(y_train)).items(), 
                               key=lambda x:x[0],reverse=False))   
              
            # compute parameters
            pi = dict() # frenquency of each  class
            mu = dict() # mean value of each  class
            '''
            Sigma: pooled sample covariance matrix
                formula: Sigma = 1/n * X_stand.T.dot(X_stand)
            '''
            X_stand = np.zeros((n,d))
            zero_pointer = 0
            for key in size_dict.keys():
                pi[key] = size_dict[key]/n
                mu[key] = np.mean(X_train[y_train==key,:],axis=0)
                # pooled sample covariance
                X_stand_k = X_train[y_train==key,:]-mu[key]
                X_stand[zero_pointer:len(X_stand_k)+zero_pointer,:] = X_stand_k
                # move the zero-pointer
                zero_pointer += len(X_stand_k)
            # pooled sample covariance matrix
            Sigma = 1/n * X_stand.T.dot(X_stand)
            
        # return, if no test dat, then return parameters
        if X_test is None:  
            return pi, mu, Sigma
        else:
            # p: probability of belonging to class k (without scaling 1)
            p = np.zeros((len(X_test), len(mu.keys()) ))
            for i, key in enumerate(mu.keys()):
                mn_k = multivariate_normal(mean=mu[key], cov = Sigma).pdf(X_test)
                # p_k: probability of belonging to class k (without scaling 1)
                p[:,i] = pi[key] * mn_k
            y_pred = np.argmax(p,axis=1)
            return y_pred
    
    ############# else, use LDA to do dimension reduction   #############
    else:
        n, d = X_train.shape
        if n_components < 1 or n_components >= d:
            raise ValueError('n_components should between 1 and d-1')
        size_dict = dict(sorted(dict(Counter(y_train)).items(), 
                               key=lambda x:x[0],reverse=False)) 
        mu = dict() # mean value of each  class
        x_bar = np.mean(X_train,axis=0)
        S_Between = np.zeros((d,d))
        S_Within = np.zeros((d,d))
        for key in size_dict.keys():  
            mu[key] = np.mean(X_train[y_train==key,:],axis=0)
            # S_between
            z = (mu[key]-x_bar).reshape(d,1)
            S_Between += z.dot(z.T)
            # S_within
            c = X_train[y_train==key,:]-mu[key]
            S_Within += c.T.dot(c)
        # get eigenvectors
        lamda, W = np.linalg.eig(np.linalg.inv(S_Within).dot(S_Between))
        # return x after dimension reduction
        X_reduct = X_train.dot(W[:,0:n_components])
        return X_reduct




###############################################################################
#---------------------------    main    -------------------------------------##

############### synthetic data generation  #############

def Cov_matrix(L,theta=np.pi/4):
  '''
  The function returns a 2x2 covariance matrix parmeterized by L and theta
  The covariance matrix is generated by R*[[L,0],[0,1]]*R.T
  where R is the rotation matrix


  '''
  Rot_matrix=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
  M=np.array([[L,0],[0,1]])
 # print("Dot product",np.dot(Rot_matrix,M))
  cov=np.dot(np.dot(Rot_matrix,M),Rot_matrix.T)
  return cov

mu={}
Sigma={}
sample={}
c={0:'purple',1:'yellow'}
sym={0:'.',1:'x'}
mu[0]=np.array([-5,0])
Sigma[0]=Cov_matrix(1,0)
mu[1]=np.array([5,0])
Sigma[1]=Sigma[0]
for k in range(len(mu)):
  #sample[k]=np.random.multivariate_normal(mu[k],Sigma[k],300)
  mn = multivariate_normal(mean=mu[k], cov=Sigma[k])
  sample[k]=mn.rvs(size=300, random_state=42)
  #sample[k]=mn.rvs(size=300) 
# data to be passed to LDA function
y0 = np.zeros((len(sample[0]),))
y1 = np.ones((len(sample[1]),))
X = np.concatenate((sample[0],sample[1]),axis=0)
y = np.concatenate((y0,y1)) 
# train test split
X_train, X_test, y_train, y_test = self_split(X,y,0.9)

###########    LDA usage   ###############################

# Usage1. train LDA classifer and get parameters
pi, mu, Sigma = self_LDA(X_train, y_train)

# Usage2. directly do prediction through LDA classifier
y_pred = self_LDA(X_train, y_train, X_test)
self_accuracy(y_pred, y_test)

# Usage3. first train the LDA and get parameters, then used the parameters to make prediction
pi, mu, Sigma = self_LDA(X_train, y_train)
y_pred = self_LDA(X_test=X_test,pi=pi, mu=mu, Sigma=Sigma)
self_accuracy(y_pred, y_test)

# Usage, use LDA to do dimension reduction
X_new = self_LDA(X_train, y_train, n_components=1)
   # plot x and y after dimension reduction
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 1, 1)
ax1.scatter(X_new,y_train)




