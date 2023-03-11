#!/usr/bin/env python
# coding: utf-8

# # Initialization

# In[1]:


import numpy as np
from matplotlib import pyplot as plt

# for p-value out of significance test
from scipy.stats import ttest_ind


# In[2]:


dataFolder = "./data"
p1 = { "testDir": dataFolder + "/p1_test.csv", "trainDir": dataFolder + "/p1_train.csv" }
p2 = { "testDir": dataFolder + "/p2_test.csv", "trainDir": dataFolder + "/p2_train.csv" }
p3 = { "testDir": dataFolder + "/p3_test.csv", "trainDir": dataFolder + "/p3_train.csv" }
p4 = {}
p5 = {}

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
    mean = np.mean(v)
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

    mse_train = mse(x_train, y_train, w)
    mae_train = mae(x_train, y_train, w)
    p_train = ttest_ind(x_train @ w, y_train).pvalue
    mse_test = mse(x_test, y_test, w)
    mae_test = mae(x_test, y_test, w)
    p_test = ttest_ind(x_test @ w, y_test).pvalue
    
    print("MSE (train-split)     : ", mse_train)
    print("MAE (train-split)     : ", mae_train)
    print("p-value (train-split) : ", p_train)

    print("--------------------------------------")

    print("a) MSE     : ", mse_test)
    print("b) MAE     : ", mae_test)
    print("c) p-value : ", p_test)
    return [mse_train, mae_train, p_train], [mse_test, mae_test, p_test]


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


p1["result"] = printResult(X, Y, X_test, Y_test)


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


p2["train"].shape, p2["test"].shape


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


p2["result"] = [[] for _ in range(8)]
p2["result"][0] = printResult(X, Y, X_test, Y_test)


# ## Non-Linear regression (1)
# $h_1(x) = w_0 + w_1x_1 + w_2x_2 + w_3x_3 + w_4x_1x_2 + w_5x_2x_3 + w_6x_3x_1 + w_7x_1^2 + w_8x_2^2 + w_9x_3^2$

# In[10]:


def makeQuadratic(data):
    n = data.shape[1]
    return np.array([data.T[i] * data.T[j] for i in range(n) for j in range(n) if j <= i]).T

p2["result"][1] = printResult(X, Y, X_test, Y_test, makeQuadratic)


# ## Non-Linear regression (2)
# $h_2(x) = w_0 + w_1x_1 + w_2x_2 + w_3x_3$
#      $+ w_4x_1x_2 + w_5x_2x_3 + w_6x_3x_1 + w_7x_1^2 + w_8x_2^2 + w_9x_3^2$
#      $+ w_{10}x_1x_2x_3 + w_{11}x_1^2x_2 + w_{12}x_2^2x_1 + w_{13}x_2^2x_3 + w_{14}x_3^2x_2 + w_{15}x_3^2x_1 + w_{16}x_1^2x_3 + w_{17}x_1^3 + w_{18}x_2^3 + w_{19}x_3^3$

# In[11]:


def makeCubic(data):
    n = data.shape[1]
    return np.array([data.T[i] * data.T[j] * data.T[k] for i in range(n) for j in range(n) for k in range(n) if j <= i and k <= j]).T

p2["result"][2] = printResult(X, Y, X_test, Y_test, makeCubic)


# ## Non-Linear regression (3)
# $h_3(x) = h_1(h_1(x))$

# In[12]:


p2["result"][3] = printResult(X, Y, X_test, Y_test, lambda x : makeQuadratic(makeQuadratic(x)))


# ## Non-Linear regression (4)
# $h_4(x) = h_1(h_2(x))$

# In[13]:


p2["result"][4] = printResult(X, Y, X_test, Y_test, lambda x : makeQuadratic(makeCubic(x)))


# ## Non-Linear regression (5)
# $h_5(x) = h_2(h_2(x))$

# In[14]:


p2["result"][5] = printResult(X, Y, X_test, Y_test, lambda x : makeCubic(makeCubic(x)))


# ## Non-Linear regression (6)
# $\phi(x) = \sum_{i=0, j=0, i \neq j}^n w_{ij} x_ie^{x_j} + \sum_{i=0}^n w_{ii} x_ie^{x_i}$
# 
# $h_6(x) = \phi(h_1(x))$

# In[15]:


def makeExp(data):
    n = data.shape[1]
    return np.array([data.T[i] * np.exp(data.T[j]) for i in range(n) for j in range(n) if i != j] + [data.T[i] * np.exp(data.T[i]) for i in range(n)]).T

p2["result"][6] = printResult(X, Y, X_test, Y_test, lambda x : makeExp(makeQuadratic(x)))


# ## Non-Linear regression (7)
# $h_7(x) = \phi(h_2(x))$

# In[16]:


p2["result"][7] = printResult(X, Y, X_test, Y_test, lambda x : makeExp(makeCubic(x)))


# In[17]:


results = np.array(p2["result"])
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot([i + 1 for i in range(8)], [np.log(row[0][0]) for row in results], label="train", marker='o')
ax[0].plot([i + 1 for i in range(8)], [np.log(row[1][0]) for row in results], label="test", marker='x')
ax[0].set_xlabel("Complexity -->")
ax[0].set_ylabel("ln(MSE)")
ax[0].legend()
ax[0].set_title("MSE vs complexity")

ax[1].plot([i + 1 for i in range(8)], [np.log(row[0][1]) for row in results], label="train", marker='o')
ax[1].plot([i + 1 for i in range(8)], [np.log(row[1][1]) for row in results], label="test", marker='x')
ax[1].set_xlabel("Complexity -->")
ax[1].set_ylabel("ln(MAE)")
ax[1].legend()
ax[1].set_title("MAE vs complexity")

ax[2].plot([i + 1 for i in range(8)], [(row[0][2]) for row in results], label="train", marker='o')
ax[2].plot([i + 1 for i in range(8)], [(row[1][2]) for row in results], label="test", marker='x')
ax[2].set_xlabel("Complexity -->")
ax[2].set_ylabel("p-value")
ax[2].legend()
ax[2].set_title("p-value vs complexity")

fig.tight_layout()


# # Custom functions for P3, P4 and P5

# In[ ]:


def logNormal(x, mean, cov):
    n = mean.shape[0]
    return - 0.5 * (n * np.log(2 * np.pi * np.linalg.det(cov)) + ((x - mean) @ np.linalg.inv(cov) @ (x - mean)))

# assume independent features
def logExp(x, mean, _):
    # ignore features with negative mean
    # for i in range(mean.size):
    #     if x[i] <= 0:
    #         np.delete(mean, i)
    #         np.delete(x, i)
    # print("Mean: ", mean)
    return - np.log(np.abs(np.prod(mean))) - np.reciprocal(mean) @ x

def naiveLogNormal(x, u, v):
    return -0.5 * np.sum([np.log(2 * np.pi * v[i][i]) + (x[i] - u[i]) * (x[i] - u[i])/v[i][i] for i in range(u.shape[0]) if v[i][i] > 0])

def classify(x, classStats, density):
    label = 0
    max = -99999
    sum = 0
    prob = []
    for key in classStats:
        mean = classStats[key]["mean"]
        cov = classStats[key]["cov"]
        prior = classStats[key]["prior"]
        value = np.log(prior) + density(x, mean, cov)
        prob.append(value)
        sum += value
        if value > max:
            max, label = value, key
    return np.r_[[label], (np.array(prob) / sum)]
    
class metrics:
    def accuracy(predicted, actual):
        m = actual.size
        correctCount = sum([1 if int(predicted[i]) == int(actual[i]) else 0 for i in range(m)])
        return correctCount / m
    
    def confusionMatrix(predicted, actual, n = 5):
        cnf = np.zeros((n, n), dtype='uint')
        for i in range(actual.size):
            cnf[int(actual[i])][int(predicted[i])] += 1
        return cnf
    
    # precision = TP/(TP + FP), recall = TP/(TP + FN), 
    # f1 = 2 * precision * recall / (precision + recall)
    def f1Score(cnf):
        sum_predict = np.sum(cnf, axis=0)
        sum_actual  = np.sum(cnf, axis=1)
        f1 = np.zeros(cnf.shape[1])
        for i in range(f1.size):
            TP = cnf[i][i]
            FP, FN = sum_predict[i] - TP, sum_actual[i] - TP
            p, r = TP/(TP + FP), TP/(TP + FN)
            f1[i] = 2 * p * r / (p + r)
        return f1
    
    def roc(predict, actual, prob, ax, labels=[0, 1], thresolds=[0, 0.1, 0.25, 0.4, 0.6, 0.8, 0.95, 1]):
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        
        for label in labels:
            tp, fp = np.zeros(len(thresolds)), np.zeros(len(thresolds))
            for t in range(len(thresolds)):
                for i in range(actual.shape[0]):
                    if float(prob[i][label]) >= thresolds[t]:
                        if actual[i] == 0:
                            tp[t] += 1
                        else:
                            fp[t] += 1
                            
            ax.plot(fp, tp, label=label, marker='x')
    
    def print(X, Y, X_test, Y_test, classStats, density):
        n_labels = len(classStats)
        train = np.array([classify(x, classStats, density) for x in X])
        test = np.array([classify(x, classStats, density) for x in X_test])
        y_train, p_train = train.T[0], train.T[1:].T
        y_test, p_test = test.T[0], test.T[1:].T
                
        cnf_train = metrics.confusionMatrix(y_train, Y, n_labels)
        cnf_test = metrics.confusionMatrix(y_test, Y_test, n_labels)
        acc_train = metrics.accuracy(y_train, Y)
        acc_test = metrics.accuracy(y_test, Y_test)
        f1_train = metrics.f1Score(cnf_train)
        f1_test = metrics.f1Score(cnf_test)
        
        print("------------------ Train ---------------------")
        print("Classification Accuracy : ", acc_train)
        print("F1 Score                : ", f1_train)
        print("------------------ Test ----------------------")
        print("Classification Accuracy : ", acc_test)
        print("F1 Score                : ", f1_test)
        
        fig, ax = plt.subplots(2, 2, figsize=(15, 8))
        ax[0][0].matshow(cnf_train)
        ax[0][0].set_xlabel("Predicted")
        ax[0][0].set_ylabel("Actual")
        ax[0][0].set_title("Confusion Matrix (train)")
        
        ax[0][1].matshow(cnf_test)
        ax[0][1].set_xlabel("Predicted")
        ax[0][1].set_ylabel("Actual")
        ax[0][1].set_title("Confusion Matrix (test)")
        
        ax[1][0].set_title("ROC (train)")
        ax[1][1].set_title("ROC (test)")
        metrics.roc(y_train, Y, p_train, ax[1][0])
        metrics.roc(y_test, Y_test, p_test, ax[1][1])

        return [acc_train, f1_train], [acc_test, f1_test]
    
def imgToFeatures(label, data, stats=False):
    X = np.array([x.flatten() for x in data]) / 255
    Y = label * np.ones(data.shape[0])
    if stats:
        return X, Y, { "mean": np.mean(X, axis=0), "cov": np.cov(X.T), "prior": data.shape[0] }
    return X, Y

# returns X, Y, X_test, Y_test and classStats
def trainTestSplit(data, train_ratio, func = imgToFeatures):
    n = data.shape[0]
    m = int(np.floor(data.shape[1] * train_ratio))
    classStats = {}
    x_train, y_train, x_test, y_test = [[[] for _ in range(n)] for _ in range(4)]
    for label in range(n):
        x_train[label], y_train[label], classStats[label] = func(label, data[label][:m], True)
        x_test[label], y_test[label] = func(label, data[label][m:])
    
    X, Y, X_test, Y_test = [x.reshape(-1, x.shape[-1]) for x in [np.array(x) for x in [x_train, y_train, x_test, y_test]]]
    return X, Y.flatten(), X_test, Y_test.flatten(), classStats


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

# In[144]:


p3["train"].shape, p3["test"].shape


# In[145]:


classStats = {}
for row in p3["train"]:
    label = int(row[-1]) - 1
    if label in classStats:
        classStats[label].append(row[:-1])
    else:
        classStats[label] = [row[:-1]]

# classStats = [np.array(data) for data in classStats]
for i in range(len(classStats)):
    data = np.array(classStats[i])
    classStats[i] = { "mean": np.mean(data, axis=0), "cov": np.cov(data.T), "prior": data.shape[0] }


# In[146]:


def splitData(data):
    X = data.T[:-1].T
    Y = data.T[-1].T.astype("int") - 1
    return X, Y

X, Y = splitData(p3["train"])
X_test, Y_test = splitData(p3["test"])


# ## Bayes' classifier with normal distribution

# In[163]:


p3["result"] = [[] for _ in range(5)]
p3["result"][0] = metrics.print(X, Y, X_test, Y_test, classStats, logNormal)


# ## Bayes' classifier with exponential distribution

# In[148]:


p3["result"][1] = metrics.print(X, Y, X_test, Y_test, classStats, logExp)


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

# In[149]:


get_ipython().run_cell_magic('capture', '', '!unzip -n data/images.zip -d data\n')


# ## Data handling

# In[150]:


import os
from os.path import join
from PIL import Image

imageDir = "./data/images"

labels = os.listdir(imageDir)
data = [[] for _ in labels]
for label in labels:
    # label = "0"
    dir = join(imageDir, label)
    files = os.listdir(dir)
    data[int(label)] = np.array([np.array(Image.open(join(dir, file)).convert("L").resize((8, 8)), dtype='uint8') for file in files])

p4["data"] = np.array(data)
p4["data"].shape


# In[151]:


fig, ax = plt.subplots(2, 5, figsize=(12, 4))
for i in range(p4["data"].shape[0]):
    ax[i // 5][i % 5].imshow(p4["data"][i][0].astype(np.uint8), cmap='gray')
    ax[i // 5][i % 5].set_title(str(i))
    ax[i // 5][i % 5].get_xaxis().set_visible(False)
    ax[i // 5][i % 5].get_yaxis().set_visible(False)

fig.tight_layout()


# In[161]:


p4["splitData"] = [trainTestSplit(p4["data"], r) for r in [0.2, 0.3, 0.5, 0.7, 0.9]]


# ## Naive Bayes

# In[162]:


p4["result"] = [[] for _ in range(5)]
p4["result"][0] = metrics.print(*p4["splitData"][0], naiveLogNormal)


# ## Logistic Regressor

# In[ ]:





# ## GMM

# In[ ]:





# # P5 (Multi-class classification)
# 
# In this part, the data from the previous problem is ’condensed’ (using PCA) to **10 dimensions**. Repeat the above experiment with all the models and metrics and record your observations.
# 
# **DATA:** `PCA_MNIST`(KannadaMNISTPCA.csv)

# In[154]:


p5["data"] = np.genfromtxt(dataFolder + "/PCA_MNIST.csv", delimiter=',')[1:]


# In[155]:


p5["data"].shape


# In[156]:


def stats(label, data, stats=False):
    X = data
    Y = label * np.ones(data.shape[0])
    if stats:
        return X, Y, { "mean": np.mean(X, axis=0), "cov": np.cov(X.T), "prior": data.shape[0] }
    return X, Y

classWiseData = [[] for _ in range(10)]
for row in p5["data"]:
    label = int(row[0])
    classWiseData[label].append(row[1:])
    
p5["splitData"] = [trainTestSplit(np.array(classWiseData), r, stats) for r in [0.2, 0.3, 0.5, 0.7, 0.9]]


# ## Naive Bayes

# In[157]:


p5["result"] = [[] for _ in range(5)]
p5["result"][0] = metrics.print(*p5["splitData"][0], naiveLogNormal)


# In[ ]:




