#!/usr/bin/env python
# coding: utf-8

# # Initialization

# In[20]:


import math
import numpy as np
from matplotlib import pyplot as plt

# for p-value out of significance test
from scipy.stats import ttest_ind


# In[2]:


dataFolder = "./data"
p1 = { "testDir": dataFolder + "/p1_test.csv", "trainDir": dataFolder + "/p1_train.csv" }
p2 = { "testDir": dataFolder + "/p2_test.csv", "trainDir": dataFolder + "/p2_train.csv" }
p3 = { "testDir": dataFolder + "/p3_test.csv", "trainDir": dataFolder + "/p3_train.csv" }

p1["test"] = np.genfromtxt(p1["testDir"], delimiter=',')
p1["train"] = np.genfromtxt(p1["trainDir"], delimiter=',')
p2["test"] = np.genfromtxt(p2["testDir"], delimiter=',')
p2["train"] = np.genfromtxt(p2["trainDir"], delimiter=',')
p3["test"] = np.genfromtxt(p3["testDir"], delimiter=',')
p3["train"] = np.genfromtxt(p3["trainDir"], delimiter=',')


# # Custom functions for P1 and P2

# In[3]:


# Compute mean squared error
def mse(X, Y, W):
    return (1/2) * (X @ W - Y) @ (X @ W - Y)

# Compute mean absolute error
def mae(X, Y, W):
    return np.sum(np.abs(X @ W - Y))

# Normalize a vector
def normalize(v):
    min = v.min()
    max = v.max()
    return (v - min) / (max - min)

# Standardize a vector
def standardize(v):
    mean = np.average(v)
    std = np.std(v)
    return (v - mean) / std

# Normalize and split the training data into features matrix with bias and the result vector
def parseData(data):
    m, n = data.shape
    data = np.array([normalize(col) for col in data.T]).T
    X = np.c_[np.ones(m), data.T[:-1].T]
    Y = data.T[-1].T
    return X, Y

# Print the required metrics
def printResult(x_train, y_train, x_test, y_test, func = lambda x : x):
    x_train = func(x_train)
    x_test = func(x_test)
    
    m, n = x_train.shape
    w = np.linalg.pinv(x_train) @ y_train

    print("MSE (train-split)     : ", mse(x_train, y_train, w))
    print("MAE (train-split)     : ", mae(x_train, y_train, w))
    print("p-value (train-split) : ", ttest_ind(x_train @ w, y_train).pvalue)

    print("--------------------------------------")

    print("a) MSE     : ", mse(x_test, y_test, w))
    print("b) MAE     : ", mae(x_test, y_test, w))
    print("c) p-value : ", ttest_ind(x_test @ w, y_test).pvalue)


# # P1 (Regression Analysis)
# 
# In this problem, the task is to predict the current health (as given by the target variable) of an organism given the measurements from two biological sensors measuring their bio-markers (negative indicates that it is lesser than the average case). 
# 
# With this data, you are expected to try our linear regression models on the  training data and report the following metrics on the test split: 
# - Mean Squared Error, 
# - Mean Absolute Error, 
# - p-value out of significance test.
# 
# **DATA:** `p1train/test.csv`

# In[4]:


p1["train"].shape


# In[5]:


X, Y = parseData(p1["train"])
X_test, Y_test = parseData(p1["test"])

# Initialise the parameters to be a null vector
W = np.array([0, 0, 0])

X, Y, W
# Check metrics with parameters as null vector
print(mse(X, Y, W))
print(mae(X, Y, W))


# ## Linear regression
# $h(x) = w_0 + w_1x_1 + w_2x_2$

# In[6]:


printResult(X, Y, X_test, Y_test)


# # P2 (Regression Analysis)
# 
# Here, you are expected to predict the lifespan of the above organism given the data from three sensors. In this case, the model is not linear.
# 
# You are expected to try several (at least 3) non-linear regression models on the train split and report the following metrics on the test split.
# - Mean Squared Error
# - Mean Absolute Error
# - p-value out of significance test
# 
# **DATA**: `p2train/test.csv`

# In[7]:


p2["train"].shape


# In[8]:


X, Y = parseData(p2["train"])
X_test, Y_test = parseData(p2["test"])

# Initialise the parameters to be a null vector
W = np.array([0, 0, 0, 0])

# Check metrics with parameters as null vector
print(mse(X, Y, W))
print(mae(X, Y, W))


# ## Linear regression
# $h(x) = w_0 + w_1x_1 + w_2x_2 + w_3x_3$

# In[9]:


printResult(X, Y, X_test, Y_test)


# ## Non-Linear regression (1)
# $h_1(x) = w_0 + w_1x_1 + w_2x_2 + w_3x_3 + w_4x_1x_2 + w_5x_2x_3 + w_6x_3x_1 + w_7x_1^2 + w_8x_2^2 + w_9x_3^2$

# In[10]:


def makeQuadratic(data):
    n = data.shape[1]
    return np.array([data.T[i] * data.T[j] for i in range(n) for j in range(n) if j <= i]).T

printResult(X, Y, X_test, Y_test, makeQuadratic)


# ## Non-Linear regression (2)
# $h_2(x) = w_0 + w_1x_1 + w_2x_2 + w_3x_3$
#      $+ w_4x_1x_2 + w_5x_2x_3 + w_6x_3x_1 + w_7x_1^2 + w_8x_2^2 + w_9x_3^2$
#      $+ w_{10}x_1x_2x_3 + w_{11}x_1^2x_2 + w_{12}x_2^2x_1 + w_{13}x_2^2x_3 + w_{14}x_3^2x_2 + w_{15}x_3^2x_1 + w_{16}x_1^2x_3 + w_{17}x_1^3 + w_{18}x_2^3 + w_{19}x_3^3$

# In[11]:


def makeCubic(data):
    n = data.shape[1]
    return np.array([data.T[i] * data.T[j] * data.T[k] for i in range(n) for j in range(n) for k in range(n) if j <= i and k <= j]).T

printResult(X, Y, X_test, Y_test, makeCubic)


# ## Non-Linear regression (3)
# $h_3(x) = h_1(h_1(x))$

# In[12]:


printResult(X, Y, X_test, Y_test, lambda x : makeQuadratic(makeQuadratic(x)))


# ## Non-Linear regression (4)
# $h_4(x) = h_1(h_2(x))$

# In[13]:


printResult(X, Y, X_test, Y_test, lambda x : makeQuadratic(makeCubic(x)))


# ## Non-Linear regression (5)
# $h_3(x) = h_2(h_2(x))$

# In[14]:


printResult(X, Y, X_test, Y_test, lambda x : makeCubic(makeCubic(x)))


# ## Non-Linear regression (6)
# $h_3(x) = h_1(h_1(h_1(x)))$

# In[15]:


printResult(X, Y, X_test, Y_test, lambda x : makeQuadratic(makeQuadratic(makeQuadratic(x))))


# ## Non-Linear regression (7)
# $h_3(x) = h_1(h_1((h_2(x)))$

# In[16]:


# printResult(X, Y, X_test, Y_test, lambda x : makeQuadratic(makeQuadratic(makeCubic(x))))


# ## Non-Linear regression (8)
# $h_3(x) = h_1(h_2(h_2(x)))$

# In[17]:


# printResult(X, Y, X_test, Y_test, lambda x : makeQuadratic(makeCubic(makeCubic(x))))


# ## Non-Linear regression (9)
# $h_3(x) = h_2(h_2(h_2(x)))$

# In[18]:


# printResult(X, Y, X_test, Y_test, lambda x : makeCubic(makeCubic(makeCubic(x))))


# # Custom functions for P3, P4 and P5

# In[19]:


# Class conditional density functions
class ccd:
    def normal(u, v):
        pass
    
    def exp(l):
        pass
    
class metrics:
    def accuracy(X, Y, W):
        return W


# # P3 (Multi-class classification)
# 
# We have data from 10 sensors fitted in an industrial plant. There are five classes indicating which product is being produced. The task is to predict the product being produced by looking at the observation from these 10 sensors. 
# 
# Given this, you are expected to implement 
# - Bayes’ classifiers with 0-1 loss assuming Normal, exponential, and GMMs (with diagonal co-variances) as class-conditional densities. For GMMs, code up the EM algorithm,
# - Linear classifier using the one-vs-rest approach
# - Multi-class Logistic regressor with gradient descent.
# 
# The metrics to be computed are 
# - Classification accuracy, 
# - Confusion matrix,
# - Class-wise F1 score, 
# - RoC curves for any pair of classes, and 
# - likelihood curve for EM with different choices for the number of mixtures as hyper-parameters, 
# - Emipiral risk on the train and test data while using logistic regressor.
# 
# **DATA:** `p3train/test.csv`

# In[23]:


p3["train"].shape


# In[21]:


math.exp(1)


# # P4 (Multi-class classification)
# 
# In this problem, we consider an image dataset called Kannada-MNIST. This dataset contains images (60,000 images with 6000 per class) of digits from the south Indian language of Kannada. The task is to build a 10-class classifier for the digits. 
# 
# You are supposed to test the following classification schemes: 
# - Naive Bayes’ with Normal as Class conditional
# - Logistic regressor with gradient descent
# - Multi-class Bayes’ classifier with GMMs with diagonal co-variances for class conditionals.
# 
# Report the following metrics on the test data: 
# - Classification accuracy
# - Confusion matrix
# - Class-wise F1 score
# - RoC curves for any pair of classes
# - likelihood curve for EM with different choices for the number of mixtures as hyper-parameters
# - Emipiral risk on the train and test data while using logistic regressor
# 
# In this problem, first split the data into train and test parts with the following ratios of **20:80**, **30:70**, **50:50**, **70:30**, and **90:10**, and record your observations. Train the algorithms on the train part and evaluate over the test part.
# 
# **DATA:** `images.zip`

# In[ ]:





# # P5 (Multi-class classification)
# 
# In this part, the data from the previous problem is ’condensed’ (using PCA) to **10 dimensions**. Repeat the above experiment with all the models and metrics and record your observations.
# 
# **DATA:** `KannadaMNISTPCA.csv`

# In[ ]:




