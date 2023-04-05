#!/usr/bin/env python
# coding: utf-8

# # Initialization

# In[2]:


import numpy as np
from matplotlib import pyplot as plt

# for p-value out of significance test
from scipy.stats import ttest_ind

# for image data handling
import os
from os.path import join, isfile, dirname
from PIL import Image

# for svm
from libsvm.svmutil import *


# # Data Handling

# ### Uncompress compressed files

# In[3]:


get_ipython().run_cell_magic('capture', '', '!unzip -n ../data/images.zip -d data')


# ### Custom functions

# In[4]:


def genFromImage(imageDir, size=(8, 8)):
    dir = dirname(imageDir)
    dataFile = join(dir, "p4_data") + str(size) + ".npy"
    if isfile(dataFile):
        with open(dataFile, 'rb') as f:
            return np.load(f)
    
    labels = os.listdir(imageDir)
    image_data = [[] for _ in labels]
    for label in labels:
        dir = join(imageDir, label)
        files = os.listdir(dir)
        image_data[int(label)] = np.array([np.array(
            Image.open(join(dir, file)).convert("L").resize(size), dtype='uint8'
        ) for file in files])
        
    image_data = np.array(image_data)
    with open(dataFile, 'wb') as f:
        np.save(f, image_data)
    return image_data

# returns X, Y, X_test, Y_test and classStats
def trainTestSplit(data, train_ratio, func):
    n = data.shape[0]
    m = int(np.floor(data.shape[1] * train_ratio))
    classStats = {}
    x_train, y_train, x_test, y_test = [[[] for _ in range(n)] for _ in range(4)]
    for label in range(n):
        x_train[label], y_train[label], classStats[label] = func(label, data[label][:m], True)
        x_test[label], y_test[label] = func(label, data[label][m:])
    
    X, Y, X_test, Y_test = [x.reshape(-1, x.shape[-1]) for x in [np.array(x) for x in [x_train, y_train, x_test, y_test]]]
    return X, Y.flatten(), X_test, Y_test.flatten(), classStats

def imgToFeatures(label, data, stats=False):
    X = np.array([x.flatten() for x in data]) / 255
    Y = label * np.ones(data.shape[0])
    if stats:
        return X, Y, { "mean": np.mean(X, axis=0), "cov": np.cov(X.T), "prior": data.shape[0], "data": X }
    return X, Y

def classify(x, classStats, density):
    label = -1
    max = -99999
    sum = 0
    prob = []
    for key in classStats:
        mean = classStats[key]["mean"]
        cov = classStats[key]["cov"]
        prior = classStats[key]["prior"]
        weights = classStats[key]["weights"] if "weights" in classStats[key] else []
        value = np.log(prior) + density(x, mean, cov, weights)
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
    
    def f1Score(cnf):
        sum_predict = np.sum(cnf, axis=0)
        sum_actual  = np.sum(cnf, axis=1)
        f1 = np.zeros(cnf.shape[1])
        for i in range(f1.size):
            TP = cnf[i][i]
            FP, FN = sum_predict[i] - TP, sum_actual[i] - TP
            p, r = TP/(TP + FP + 1e-8), TP/(TP + FN + 1e-8)
            f1[i] = 2 * p * r / (p + r + 1e-8)
        return f1
    
    def print(X, Y, X_test, Y_test, classStats, density, result=True):
        n_labels = len(classStats)
        train = np.array([classify(x, classStats, density) for x in X])
        test = np.array([classify(x, classStats, density) for x in X_test])
        # train = classify(X, classStats, density)
        # test = classify(X, classStats, density)
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

        if result:
            return [acc_train, f1_train], [acc_test, f1_test]


# ### Data extraction

# In[5]:


dataFolder = "../data"
imageDir = join(dataFolder, "images")
imageDataDir = join(dataFolder, "p4_data.csv")

p1 = { "testDir": dataFolder + "/p1_test.csv", "trainDir": dataFolder + "/p1_train.csv" } # regression
p2 = { "testDir": dataFolder + "/p2_test.csv", "trainDir": dataFolder + "/p2_train.csv" } # regression
p3 = { "testDir": dataFolder + "/p3_test.csv", "trainDir": dataFolder + "/p3_train.csv" } # classification
p4 = {}                                                                                   # classification
p5 = {}                                                                                   # classification

p1["test"] = np.genfromtxt(p1["testDir"], delimiter=',')
p1["train"] = np.genfromtxt(p1["trainDir"], delimiter=',')
p2["test"] = np.genfromtxt(p2["testDir"], delimiter=',')
p2["train"] = np.genfromtxt(p2["trainDir"], delimiter=',')
p3["test"] = np.genfromtxt(p3["testDir"], delimiter=',')
p3["train"] = np.genfromtxt(p3["trainDir"], delimiter=',')
p4["data"] = genFromImage(imageDir)
p5["data"] = np.genfromtxt(dataFolder + "/PCA_MNIST.csv", delimiter=',')[1:]

print("--------------------------- Data Shapes ------------------------------")
print("    (Regression) p1[train]:      ", p1["train"].shape, ", p1[test]: ", p1["test"].shape)
print("    (Regression) p2[train]:      ", p2["train"].shape, ", p2[test]: ", p2["test"].shape)
print("(Classification) p3[train]:     ", p3["train"].shape, ", p3[test]: ", p3["test"].shape)
print("(Classification)  p4[data]:", p4["data"].shape)
print("(Classification)  p5[data]:     ", p5["data"].shape)


# In[6]:


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
    classStats[i] = { "mean": np.mean(data, axis=0), "cov": np.cov(data.T), "prior": data.shape[0], "data": data }
    
def splitData(data):
    # X = np.array([normalize(col) for col in data.T[:-1]]).T
    X = data.T[:-1].T
    Y = data.T[-1].T.astype("int") - 1
    return X, Y

p3["classStats"] = classStats
p3["X"], p3["Y"] = splitData(p3["train"])
p3["X_test"], p3["Y_test"] = splitData(p3["test"])

p3["X"].shape, p3["Y"].shape, p3["X_test"].shape, p3["Y_test"].shape


# # Custom functions for SVM

# In[7]:


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
    # data = np.array([normalize(col) for col in data.T]).T
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


# # P1 (SVM)
# 
# - For the classification problems given in A1, implement SVMs both with and without slack formulations. 
# - Experiment with at least 3 Kernels and grid search on hyper-parameters on different kernels.
# - Report your observations (you can use standard python library, LibSVM and need not implement SMO). 
# - For multi-class classification, implement a one-vs-rest approach.
# 
# **DATA:** `p3train/test.csv`, `images.zip (p4[data])`

# In[28]:


np.unique(Y)


# In[10]:


# https://www.csie.ntu.edu.tw/~cjlin/libsvm/
# https://github.com/prathmachowksey/Fisher-Linear-Discriminant-Analysis/blob/master/fisher_lda.ipynb
# https://python-course.eu/machine-learning/linear-discriminant-analysis-in-python.php

X, Y = parseData(p3["train"])
X_test, Y_test = parseData(p3["test"])
Y, Y_test = Y - 1, Y_test - 1
labels = np.unique(Y)

# Initialise the parameters to be a null vector
W = svm_train(Y, X)
p_label, p_acc, p_val = svm_predict(Y_test, X_test, W)

p_acc


# In[57]:


X, Y = p3["train"][:, :-1], p3["train"][:, -1]
X_test, Y_test = p3["test"][:, :-1], p3["test"][:, -1]

# train one-vs-rest approach
svm = {}
for k in range(len(labels)):
    svm[k] = svm_train(np.double(Y==k+1), X, '-c 1 -g 0.2 -b 1')


# In[83]:


prob = np.zeros((len(Y_test), len(labels)));

for k in range(len(labels)):
    _, _, p = svm_predict(np.double(Y_test==k), X_test, svm[k], '-b 1')
    prob[:,k] = np.array(p)[:, 1]
    
pred = np.argmax(prob, axis=1) + 1;
acc = np.sum(np.abs(pred - Y_test) < 0.5) / len(Y_test)
print("Overall Accuracy: ", acc)


# # P2 (SVM)
# 
# Implement FLDA for the classification problems in A1 and report the metrics as in A1
# 
# **DATA**: `p3train/test.csv`, `images.zip (p4[data])`

# In[55]:


X, Y = p3["train"][:, :-1], p3["train"][:, -1]
X_test, Y_test = p3["test"][:, :-1], p3["test"][:, -1]

def solveFLDA(data, k):
    x1, x2 = data[data[:, -1] == k][:, :-1], data[data[:, -1] != k][:, :-1]
    mean1 = np.mean(x1,axis=0)
    mean2 = np.mean(x2,axis=0)
    mean_diff = np.subtract(mean1,mean2)

    cov1 = np.cov(np.transpose(x1))
    cov2 = np.cov(np.transpose(x2))

    # within class spread 
    SW = np.add(cov1,cov2)
    
    # w = (SW^{-1}).(M1-M2)
    w = np.dot(np.linalg.inv(SW), mean_diff)
    
    # normalise W
    w_norm = w / np.linalg.norm(w)
    
    return w_norm

# train one-vs-rest approach
flda = {}
for k in range(len(labels)):
    flda[k] = solveFLDA(p3["train"], k+1);

# get probability estimates of test instances using each model
prob = np.zeros((len(Y_test), len(labels)));
for k in range(len(labels)):
    prob[:, k] = X_test @ flda[k]

# predict the class with the highest probability
pred = np.argmax(prob, axis=1);
acc = np.sum((pred+1) == Y_test) / len(Y_test)    # accuracy

acc
# Check metrics with parameters as null vector
# print(mse(X, Y, W))
# print(mae(X, Y, W))


# # P3 (SVM)
# 
# For the regression problem p1 in A1, overfit the data with over-parameterized models (at least 3). 
# In the next part, impose different types of regularizers (L2, L1, and a combination of both) and 
# plot the bias-variance curves.
# 
# **DATA**: `p1train/test.csv`

# # Custom functions for Neural Networks

# In[ ]:


max_float = np.finfo("float64").max
max_exp = np.log(max_float)

def normal(x, mean ,cov, *args):
    n = len(mean)
    val = -0.5 * (x - mean) @ np.linalg.pinv(cov) @ (x - mean)
    return np.exp(val) if np.abs(val) < max_exp else (1/max_float) / ((2 * np.pi * np.linalg.det(cov)) ** (n/2) + 1e-8)

def logGMM(x, mean, cov, weights, *args):
    k = len(weights)
    val = np.log(np.sum([weights[i] * normal(x, mean[i], cov[i]) for i in range(k)]) + 1e-8)
    return val

def logNormal(x, mean, cov, *args):
    n = mean.shape[0]
    return - 0.5 * (n * np.log(2 * np.pi * np.linalg.det(cov)) + ((x - mean) @ np.linalg.inv(cov) @ (x - mean).T))

# assume independent features
def logExp(x, mean, *args):
    return - np.log(np.abs(np.prod(mean))) - np.reciprocal(mean) @ x

def naiveLogNormal(x, u, v, *args):
    return -0.5 * np.sum([np.log(2 * np.pi * v[i][i]) + (x[i] - u[i]) * (x[i] - u[i])/v[i][i] for i in range(u.shape[0]) if v[i][i] > 0])

def classify(x, classStats, density):
    label = -1
    max = -99999
    sum = 0
    prob = []
    for key in classStats:
        mean = classStats[key]["mean"]
        cov = classStats[key]["cov"]
        prior = classStats[key]["prior"]
        weights = classStats[key]["weights"] if "weights" in classStats[key] else []
        value = np.log(prior) + density(x, mean, cov, weights)
        prob.append(value)
        sum += value
        if value > max:
            max, label = value, key
    return np.r_[[label], (np.array(prob) / sum)]


# In[ ]:


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
    
    def f1Score(cnf):
        sum_predict = np.sum(cnf, axis=0)
        sum_actual  = np.sum(cnf, axis=1)
        f1 = np.zeros(cnf.shape[1])
        for i in range(f1.size):
            TP = cnf[i][i]
            FP, FN = sum_predict[i] - TP, sum_actual[i] - TP
            p, r = TP/(TP + FP + 1e-8), TP/(TP + FN + 1e-8)
            f1[i] = 2 * p * r / (p + r + 1e-8)
        return f1
    
    def roc(predict, actual, prob, ax, labels=[0, 1], thresolds=[0, 0.2, 0.4, 0.6, 0.8, 1]):
        for label in labels:
            tp, fp, tn, fn = [np.zeros(len(thresolds)) for _ in range(4)]
            for t in range(len(thresolds)):
                for i in range(actual.shape[0]):
                    if float(prob[i][label]) >= thresolds[t]:
                        if actual[i] == 0:
                            tp[t] += 1.0
                        else:
                            fp[t] += 1.0
                    else:
                        if actual[i] == 0:
                            fn[t] += 1.0
                        else:
                            tn[t] += 1.0
                        
            fpr = fp / (fp + tn + 1e-8)
            tpr = tp / (tp + fn + 1e-8)
            ax.plot(fpr, tpr, label=label, marker='x')        
        
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend()
        
    
    def print(X, Y, X_test, Y_test, classStats, density, result=True):
        n_labels = len(classStats)
        train = np.array([classify(x, classStats, density) for x in X])
        test = np.array([classify(x, classStats, density) for x in X_test])
        # train = classify(X, classStats, density)
        # test = classify(X, classStats, density)
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
        # print("Confusion Matrix        : ")
        # print(cnf_test)
        
        fig, ax = plt.subplots(2, 2, figsize=(16, 16))
        ax[0][0].matshow(cnf_train.T, cmap='GnBu')
        ax[0][0].set_xlabel("Predicted")
        ax[0][0].set_ylabel("Actual")
        ax[0][0].set_title("Confusion Matrix (train)")
        for (x, y), value in np.ndenumerate(cnf_train):
            ax[0][0].text(x, y, f"{value: d}", va="center", ha="center")
        
        ax[0][1].matshow(cnf_test.T, cmap='GnBu')
        ax[0][1].set_xlabel("Predicted")
        ax[0][1].set_ylabel("Actual")
        ax[0][1].set_title("Confusion Matrix (test)")
        for (x, y), value in np.ndenumerate(cnf_test):
            ax[0][1].text(x, y, f"{value: d}", va="center", ha="center")
        
        thresolds = [i/100 for i in range(100)]
        metrics.roc(y_train, Y, p_train, ax[1][0], thresolds=thresolds)
        metrics.roc(y_test, Y_test, p_test, ax[1][1], thresolds=thresolds)
        ax[1][0].set_title("ROC (train)")
        ax[1][1].set_title("ROC (test)")
        
        if result:
            return [acc_train, f1_train], [acc_test, f1_test]


# # P4 (Neural Networks, MLP)
# 
# - Construct a Multi-layer Perception (MLP) or a feed-forward neural network to work on the K-MNIST dataset. 
# - Experiment with at least 3 settings of the number of hidden layers and Neurons. 
# - Explicitly code the Error Backpropagation algorithm as a class and use it on MLPs with different architectures and loss functions (CE, squared error loss).
# - For this part, you should only use Numpy. 
# 
# Report the accuracy and F1 scores with all the considered configurations.
# 
# **DATA:** `p3train/test.csv`

# ## Data Handling

# In[ ]:


p3["train"].shape, p3["test"].shape


# In[ ]:


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
    classStats[i] = { "mean": np.mean(data, axis=0), "cov": np.cov(data.T), "prior": data.shape[0], "data": data }


# In[ ]:


def splitData(data):
    # X = np.array([normalize(col) for col in data.T[:-1]]).T
    X = data.T[:-1].T
    Y = data.T[-1].T.astype("int") - 1
    return X, Y

X, Y = splitData(p3["train"])
X_test, Y_test = splitData(p3["test"])

X.shape, Y.shape, X_test.shape, Y_test.shape


# ## Bayes' classifier with normal distribution

# In[ ]:


p3["result"] = [[] for _ in range(5)]
p3["result"][0] = metrics.print(X, Y, X_test, Y_test, classStats, logNormal)


# ## Bayes' classifier with exponential distribution

# In[ ]:


p3["result"][1] = metrics.print(X, Y, X_test, Y_test, classStats, logExp)


# ## Bayes' classifier with GMM distribution

# In[ ]:


def printGmmP3(number_of_guassians , max_iter = 50):
    classStatsGMM = {}
    for label in classStats:
        classStatsGMM[label] = { "prior": classStats[label]["prior"] }
        classStatsGMM[label]["weights"], classStatsGMM[label]["mean"], classStatsGMM[label]["cov"] = em(classStats[label]["data"], number_of_guassians, max_iter)
        print("weights of class ", str(label + 1), ": ", classStatsGMM[label]["weights"])

    metrics.print(X, Y, X_test, Y_test, classStatsGMM, logGMM, result=False)


# In[ ]:


printGmmP3(2)


# In[ ]:


printGmmP3(5)


# In[ ]:


printGmmP3(8)


# ## Logistic Regression

# In[ ]:


train_data = p3["train"]
test_data = p3["test"]

# Split data into features and labels
X_train = train_data[:, :-1]
y_train_orig = train_data[:, -1]
X_test = test_data[:, :-1]
y_test_orig = test_data[:, -1]

# One-hot encode target variable
num_classes = 5
num_samples = y_train_orig.shape[0]
y_train = np.zeros((num_samples, num_classes))
for i in range(num_samples):
    y_train[i, int(y_train_orig[i]) - 1] = 1


# Define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define softmax function
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Initialize weights and biases
num_features = X_train.shape[1]
W = np.random.randn(num_features, num_classes)
b = np.random.randn(num_classes)

# Set hyperparameters
learning_rate = 0.1
num_iterations = 1000
epsilon = 1e-8

# Train model using gradient descent
prev_loss = float('inf')
for i in range(num_iterations):
    # Forward propagation
    z = np.dot(X_train, W) + b
    y_pred = softmax(z)

    # Compute loss
    loss = -np.sum(y_train * np.log(y_pred + epsilon)) / num_samples

    # Backward propagation
    dz = y_pred - y_train
    dW = np.dot(X_train.T, dz) / num_samples
    db = np.sum(dz, axis=0) / num_samples

    # Update weights and biases
    W -= learning_rate * dW
    b -= learning_rate * db

    # Check stopping criterion
    if prev_loss - loss < epsilon:
        print('Stopping criterion met')
        break

    prev_loss = loss

# Evaluate model on test set
z = np.dot(X_test, W) + b
y_pred = np.argmax(softmax(z), axis=1) + 1
accuracy = np.sum(y_pred == y_test_orig) / y_test_orig.shape[0]
print('Test accuracy:', accuracy)


z_train = np.dot(X_train, W) + b
y_train_pred = np.argmax(softmax(z_train), axis=1) + 1
train_loss = -np.sum(y_train * np.log(softmax(z_train) + epsilon)) / num_samples
train_error_rate = 1 - np.sum(y_train_pred == y_train_orig) / y_train_orig.shape[0]
print('Training empirical risk:', train_loss)
print('Training error rate:', train_error_rate)

# Compute empirical risk on test data
num_samples_test = y_test_orig.shape[0]
y_test = np.zeros((num_samples_test, num_classes))
for i in range(num_samples_test):
    y_test[i, int(y_test_orig[i]) - 1] = 1

z_test = np.dot(X_test, W) + b
test_loss = -np.sum(y_test * np.log(softmax(z_test) + epsilon)) / num_samples_test
test_error_rate = 1 - np.sum(y_pred == y_test_orig) / y_test_orig.shape[0]
print('Test empirical risk:', test_loss)
print('Test error rate:', test_error_rate)



num_classes = len(np.unique(y_test_orig))
confusion_matrix = np.zeros((num_classes, num_classes))
for i in range(len(y_test_orig)):
    true_class = int(y_test_orig[i] - 1)
    predicted_class = int(y_pred[i] - 1)
    confusion_matrix[true_class, predicted_class] += 1
# print('Confusion matrix:')
# print(confusion_matrix)


num_classes = len(np.unique(y_test_orig))
f1_scores = np.zeros(num_classes)
for i in range(num_classes):
    true_positives = confusion_matrix[i, i]
    false_positives = np.sum(confusion_matrix[:, i]) - true_positives
    false_negatives = np.sum(confusion_matrix[i, :]) - true_positives
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1_scores[i] = 2 * precision * recall / (precision + recall + 1e-8)
print('Class-wise F1 score:')
print(f1_scores)

# from matplotlib import pyplot as plt

# # Choose two classes
# class_1 = 5
# class_2 = 3

# # Get predicted probabilities for the two classes
# y_class_1 = y_pred == class_1
# y_class_2 = y_pred == class_2
# y_prob_1 = softmax(z)[:, class_1 - 1]
# y_prob_2 = softmax(z)[:, class_2 - 1]

# # Compute true positive rate and false positive rate
# num_thresholds = 100
# tpr = np.zeros(num_thresholds)
# fpr = np.zeros(num_thresholds)
# for i in range(num_thresholds):
#     threshold = i / (num_thresholds - 1)
#     tp = np.sum((y_prob_1 >= threshold) & (y_class_1 == True))
#     fp = np.sum((y_prob_1 >= threshold) & (y_class_1 == False))
#     tn = np.sum((y_prob_2 < threshold) & (y_class_2 == True))
#     fn = np.sum((y_prob_2 < threshold) & (y_class_2 == False))
#     tpr[i] = tp / (tp + fn + 1e-8)
#     fpr[i] = fp / (fp + tn + 1e-8)

# # Plot RoC curve and confusion matrix
# fig, ax = plt.subplots(2, 1, figsize=(8, 16))
# ax[0].matshow(confusion_matrix, cmap='GnBu')
# ax[0].set_xlabel("Predicted")
# ax[0].set_ylabel("Actual")
# ax[0].set_title("Confusion Matrix")
# for (x, y), value in np.ndenumerate(confusion_matrix):
#     ax[0].text(x, y, f"{value: .0f}", va="center", ha="center")

# ax[1].plot(fpr, tpr, marker='x')
# ax[1].set_xlabel("False positive rate")
# ax[1].set_ylabel("True positive rate")                     
# ax[1].set_title("ROC curve for classes {} and {}".format(class_1, class_2))

# fig.tight_layout()

# Choose two classes
class_1 = 1
class_2 = 2
# Get predicted probabilities for the two classes
y_class_1 = y_pred == class_1
y_class_2 = y_pred == class_2
y_prob_1 = softmax(z)[:, class_1 - 1]
y_prob_2 = softmax(z)[:, class_2 - 1]

# Compute true positive rate and false positive rate for both classes
num_thresholds = 100
tpr_class_1 = np.zeros(num_thresholds)
fpr_class_1 = np.zeros(num_thresholds)
tpr_class_2 = np.zeros(num_thresholds)
fpr_class_2 = np.zeros(num_thresholds)

for i in range(num_thresholds):
    threshold = i / (num_thresholds - 1)
    tp_class_1 = np.sum((y_prob_1 >= threshold) & (y_class_1 == True))
    fn_class_1 = np.sum((y_prob_1 < threshold) & (y_class_1 == True))
    tn_class_1 = np.sum((y_prob_2 < threshold) & (y_class_2 == True))
    fp_class_1 = np.sum((y_prob_2 >= threshold) & (y_class_2 == False))
    tpr_class_1[i] = tp_class_1 / (tp_class_1 + fn_class_1 + 1e-8)
    fpr_class_1[i] = fp_class_1 / (fp_class_1 + tn_class_1 + 1e-8)
    
    tp_class_2 = np.sum((y_prob_2 >= threshold) & (y_class_2 == True))
    fn_class_2 = np.sum((y_prob_2 < threshold) & (y_class_2 == True))
    tn_class_2 = np.sum((y_prob_1 < threshold) & (y_class_1 == True))
    fp_class_2 = np.sum((y_prob_1 >= threshold) & (y_class_1 == False))
    tpr_class_2[i] = tp_class_2 / (tp_class_2 + fn_class_2 + 1e-8)
    fpr_class_2[i] = fp_class_2 / (fp_class_2 + tn_class_2 + 1e-8)

# Plot RoC curves and confusion matrix
fig, ax = plt.subplots(2, 2, figsize=(12, 12))
ax[0, 0].matshow(confusion_matrix, cmap='GnBu')
ax[0, 0].set_xlabel("Predicted")
ax[0, 0].set_ylabel("Actual")
ax[0, 0].set_title("Confusion Matrix")
for (x, y), value in np.ndenumerate(confusion_matrix):
    ax[0, 0].text(x, y, f"{value: .0f}", va="center", ha="center")

ax[0, 1].plot(fpr_class_1, tpr_class_1, marker='x')
ax[0, 1].set_xlabel("False positive rate")
ax[0, 1].set_ylabel("True positive rate")                     
ax[0, 1].set_title("ROC curve for class {}".format(class_1))

ax[1, 0].plot(fpr_class_2, tpr_class_2, marker='x')
ax[1, 0].set_xlabel("False positive rate")
ax[1, 0].set_ylabel("True Positive rate")
ax[1, 0].set_title("ROC curve for class {}".format(class_2))

ax[1, 1].plot(fpr_class_1, tpr_class_1, marker='x', label="Class {}".format(class_1))
ax[1, 1].plot(fpr_class_2, tpr_class_2, marker='o', label="Class {}".format(class_2))
ax[1, 1].set_xlabel("False positive rate")
ax[1, 1].set_ylabel("True positive rate")
ax[1, 1].set_title("ROC curve for classes {} and {}".format(class_1, class_2))
ax[1, 1].legend()

fig.tight_layout()
plt.show()


# ## Linear classifier using one vs all approach

# In[ ]:


data = p3["train"]
X = data[:, :-1]  # Features
y = data[:, -1]   # Labels

# One-hot encode target variable
num_classes = 5
num_samples = y.shape[0]
y_encoded = np.zeros((num_samples, num_classes))
for i in range(num_samples):
    y_encoded[i, int(y[i]) - 1] = 1

# Add a column of 1s to X for bias term
X = np.hstack((X, np.ones((num_samples, 1))))

# Initialize weights
num_features = X.shape[1]
W = np.random.randn(num_features, num_classes)

# Set hyperparameters
learning_rate = 0.01
num_iterations = 1000
epsilon = 1e-8

# Train model using gradient descent
prev_loss = float('inf')
for i in range(num_iterations):
    # Forward propagation
    z = np.dot(X, W)
    y_pred = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    # Compute loss
    loss = -np.sum(y_encoded * np.log(y_pred + epsilon)) / num_samples

    # Backward propagation
    dz = y_pred - y_encoded
    dW = np.dot(X.T, dz) / num_samples

    # Update weights
    W -= learning_rate * dW

    # Check stopping criterion
    if prev_loss - loss < epsilon:
        print('Stopping criterion met')
        break

    prev_loss = loss

# Evaluate model on test set
X_test = test_data[:, :-1]
y_test_orig = test_data[:, -1]
num_test_samples = y_test_orig.shape[0]

# One-hot encode target variable
y_test = np.zeros((num_test_samples, num_classes))
for i in range(num_test_samples):
    y_test[i, int(y_test_orig[i]) - 1] = 1

# Add a column of 1s to X_test for bias term
X_test = np.hstack((X_test, np.ones((num_test_samples, 1))))

# Compute predictions on test set
z_test = np.dot(X_test, W)
y_test_pred = np.argmax(z_test, axis=1) + 1

# Compute test accuracy
test_accuracy = np.sum(y_test_pred == y_test_orig) / num_test_samples
print('Test accuracy:', test_accuracy)

conf_matrix = np.zeros((num_classes, num_classes))
for i in range(num_test_samples):
    true_class = int(y_test_orig[i]) - 1
    pred_class = int(y_test_pred[i]) - 1
    conf_matrix[true_class, pred_class] += 1
print('Confusion matrix:')
print(conf_matrix)

# Compute class-wise F1 score
f1_scores = []
for c in range(num_classes):
    tp = conf_matrix[c,c]
    fp = np.sum(conf_matrix[:,c]) - tp
    fn = np.sum(conf_matrix[c,:]) - tp
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    f1_scores.append(f1)
print('Class-wise F1 score:', f1_scores)

# Compute predictions on test set
z_test = np.dot(X_test, W)
y_test_prob = np.exp(z_test) / np.sum(np.exp(z_test), axis=1, keepdims=True)
y_test_pred = np.argmax(z_test, axis=1) + 1

# # Choose two classes for ROC curve
# class1 = 1
# class2 = 2

# # Compute false positive rate and true positive rate for different thresholds
# fpr = []
# tpr = []
# num_thresholds = 100
# for i in range(num_thresholds):
#     threshold = i / num_thresholds
#     tp = 0
#     fp = 0
#     tn = 0
#     fn = 0
#     for j in range(num_test_samples):
#         if y_test_orig[j] == class1:
#             if y_test_prob[j][class1-1] >= threshold:
#                 tp += 1
#             else:
#                 fn += 1
#         elif y_test_orig[j] == class2:
#             if y_test_prob[j][class1-1] >= threshold:
#                 fp += 1
#             else:
#                 tn += 1
#     fpr.append(fp / (fp + tn))
#     tpr.append(tp / (tp + fn))

# # Plot RoC curve and confusion matrix
# fig, ax = plt.subplots(2, 1, figsize=(8, 16))
# ax[0].matshow(confusion_matrix, cmap='GnBu')
# ax[0].set_xlabel("Predicted")
# ax[0].set_ylabel("Actual")
# ax[0].set_title("Confusion Matrix")
# for (x, y), value in np.ndenumerate(confusion_matrix):
#     ax[0].text(x, y, f"{value: .0f}", va="center", ha="center")

# ax[1].plot(fpr, tpr, marker='x')
# ax[1].set_xlabel("False positive rate")
# ax[1].set_ylabel("True positive rate")                     
# ax[1].set_title("ROC curve for classes {} and {}".format(class_1, class_2))

# fig.tight_layout()
# Choose two classes for ROC curve
class1 = 1
class2 = 2

# Choose two classes for ROC curve
class1 = 1
class2 = 2

# Compute false positive rate and true positive rate for different thresholds
fpr_class1 = []
tpr_class1 = []
fpr_class2 = []
tpr_class2 = []
num_thresholds = 100
for i in range(num_thresholds):
    threshold = i / num_thresholds
    tp_class1 = 0
    fp_class1 = 0
    tn_class1 = 0
    fn_class1 = 0
    tp_class2 = 0
    fp_class2 = 0
    tn_class2 = 0
    fn_class2 = 0
    for j in range(num_test_samples):
        if y_test_orig[j] == class1:
            if y_test_prob[j][class1-1] >= threshold:
                tp_class1 += 1
            else:
                fn_class1 += 1
        elif y_test_orig[j] == class2:
            if y_test_prob[j][class1-1] >= threshold:
                fp_class2 += 1
            else:
                tn_class2 += 1
        if y_test_orig[j] == class2:
            if y_test_prob[j][class2-1] >= threshold:
                tp_class2 += 1
            else:
                fn_class2 += 1
        elif y_test_orig[j] == class1:
            if y_test_prob[j][class2-1] >= threshold:
                fp_class1 += 1
            else:
                tn_class1 += 1
    fpr_class1.append(fp_class1 / (fp_class1 + tn_class1))
    tpr_class1.append(tp_class1 / (tp_class1 + fn_class1))
    fpr_class2.append(fp_class2 / (fp_class2 + tn_class2))
    tpr_class2.append(tp_class2 / (tp_class2 + fn_class2))

# Plot RoC curves and confusion matrix
fig, ax = plt.subplots(2, 2, figsize=(12, 12))
ax[0, 0].matshow(confusion_matrix, cmap='GnBu')
ax[0, 0].set_xlabel("Predicted")
ax[0, 0].set_ylabel("Actual")
ax[0, 0].set_title("Confusion Matrix")
for (x, y), value in np.ndenumerate(confusion_matrix):
    ax[0, 0].text(x, y, f"{value: .0f}", va="center", ha="center")

ax[0, 1].plot(fpr_class1, tpr_class1, marker='x')
ax[0, 1].set_xlabel("False positive rate")
ax[0, 1].set_ylabel("True positive rate")                     
ax[0, 1].set_title("ROC curve for class {}".format(class1))

ax[1, 0].plot(fpr_class2, tpr_class2, marker='x')
ax[1, 0].set_xlabel("False positive rate")
ax[1, 0].set_ylabel("True positive rate")                     
ax[1, 0].set_title("ROC curve for class {}".format(class2))

ax[1, 1].plot(fpr_class1, tpr_class1, marker='x', label=f"class {class1}")
ax[1, 1].plot(fpr_class2, tpr_class2, marker='o', label=f"class {class2}")
ax[1, 1].set_xlabel("False positive rate")
ax[1, 1].set_ylabel("True positive rate")
ax[1, 1].set_title("ROC curve for classes {} and {}".format(class1, class2))
ax[1, 1].legend()
plt.show()


# # P5 (Neural Networks, CNN)
# 
# - Construct a CNN for the K-MNIST dataset and code the back-propagation algorithm with weight sharing and local-receptive fields. 
# - Experiment with 3 different architectures and report the accuracy.
# 
# **DATA:** `images.zip (p4[data])`

# # P6 (Neural Networks, CNN)
# 
# - For the above problem, build a big-enough CNN architecture that would overfit the K-MNIST data. 
# - Impose L2 and early-stopping as regularizers and plot the bias-variance curves. 
# - Perturb each of the input images with additive Gaussian noise and report its regularization impact.
# 
# **DATA:** `images.zip (p4[data])`

# ## Data handling

# In[ ]:


p4["splitData"] = [trainTestSplit(p4["data"], r, imgToFeatures) for r in [0.2, 0.3, 0.5, 0.7, 0.9]]


# ## Naive Bayes

# In[ ]:


p4["result"] = [[] for _ in range(5)]


# ### Test split -- 20:80

# In[ ]:


p4["result"][0] = metrics.print(*p4["splitData"][0], naiveLogNormal)


# ### Test split -- 30:70

# In[ ]:


p4["result"][0] = metrics.print(*p4["splitData"][1], naiveLogNormal)


# ### Test split -- 50:50

# In[ ]:


p4["result"][0] = metrics.print(*p4["splitData"][2], naiveLogNormal)


# ### Test split -- 70:30

# In[ ]:


p4["result"][0] = metrics.print(*p4["splitData"][3], naiveLogNormal)


# ### Test split -- 90:10

# In[ ]:


p4["result"][0] = metrics.print(*p4["splitData"][4], naiveLogNormal)


# ## GMM

# In[ ]:


def printGmm(data, number_of_guassians=2):
    classStatsGMM = {}
    for label in data[-1]:
        classStatsGMM[label] = { "prior": data[-1][label]["prior"] }
        classStatsGMM[label]["weights"], classStatsGMM[label]["mean"], classStatsGMM[label]["cov"] = em(data[-1][label]["data"], number_of_guassians, 50)

    metrics.print(*data[:-1], classStatsGMM, logGMM, result=False)


# ### Test split -- 20:80

# In[ ]:


printGmm(p4["splitData"][0])


# ### Test split -- 30:70

# In[ ]:


printGmm(p4["splitData"][1])


# ### Test split -- 50:50

# In[ ]:


printGmm(p4["splitData"][2])


# ### Test split -- 70:30

# In[ ]:


printGmm(p4["splitData"][3])


# ### Test split -- 90:10

# In[ ]:


printGmm(p4["splitData"][4])


# ## Logistic Regression

# In[ ]:


def logisticRegressor(data):
    X_train,y_train_orig , X_test, y_test_orig, classStats = data
    num_classes = 10
    num_samples = y_train_orig.shape[0]
    y_train = np.zeros((num_samples, num_classes))
    for i in range(num_samples):
        y_train[i, int(y_train_orig[i]) - 1] = 1

    # Define sigmoid function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Define softmax function
    def softmax(x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # Initialize weights and biases
    num_features = X_train.shape[1]
    W = np.random.randn(num_features, num_classes)
    b = np.random.randn(num_classes)

    # Set hyperparameters
    learning_rate = 0.1
    num_iterations = 1000
    epsilon = 1e-8

    # Train model using gradient descent
    prev_loss = float('inf')
    for i in range(num_iterations):
        # Forward propagation
        z = np.dot(X_train, W) + b
        y_pred = softmax(z)

        # Compute loss
        loss = -np.sum(y_train * np.log(y_pred + epsilon)) / num_samples

        # Backward propagation
        dz = y_pred - y_train
        dW = np.dot(X_train.T, dz) / num_samples
        db = np.sum(dz, axis=0) / num_samples

        # Update weights and biases
        W -= learning_rate * dW
        b -= learning_rate * db

        # Check stopping criterion
        if prev_loss - loss < epsilon:
            print('Stopping criterion met')
            break

        prev_loss = loss

    # Evaluate model on test set
    z = np.dot(X_test, W) + b
    y_pred = np.argmax(softmax(z), axis=1) + 1
    accuracy = np.sum(y_pred == y_test_orig) / y_test_orig.shape[0]
    print('Test accuracy:', accuracy)

    z_train = np.dot(X_train, W) + b
    y_train_pred = np.argmax(softmax(z_train), axis=1) + 1
    train_loss = -np.sum(y_train * np.log(softmax(z_train) + epsilon)) / num_samples
    train_error_rate = 1 - np.sum(y_train_pred == y_train_orig) / y_train_orig.shape[0]
    print('Training empirical risk:', train_loss)
    print('Training error rate:', train_error_rate)

    # Compute empirical risk on test data
    num_samples_test = y_test_orig.shape[0]
    y_test = np.zeros((num_samples_test, num_classes))
    for i in range(num_samples_test):
        y_test[i, int(y_test_orig[i]) - 1] = 1

    z_test = np.dot(X_test, W) + b
    test_loss = -np.sum(y_test * np.log(softmax(z_test) + epsilon)) / num_samples_test
    test_error_rate = 1 - np.sum(y_pred == y_test_orig) / y_test_orig.shape[0]
    print('Test empirical risk:', test_loss)
    print('Test error rate:', test_error_rate)

    num_classes = len(np.unique(y_test_orig))
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_test_orig)):
        true_class = int(y_test_orig[i] - 1)
        predicted_class = int(y_pred[i] - 1)
        confusion_matrix[true_class, predicted_class] += 1
    # print('Confusion matrix:')
    # print(confusion_matrix)


    num_classes = len(np.unique(y_test_orig))
    f1_scores = np.zeros(num_classes)
    for i in range(num_classes):
        true_positives = confusion_matrix[i, i]
        false_positives = np.sum(confusion_matrix[:, i]) - true_positives
        false_negatives = np.sum(confusion_matrix[i, :]) - true_positives
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1_scores[i] = 2 * precision * recall / (precision + recall + 1e-8)
    print('Class-wise F1 score:')
    print(f1_scores)

    # Choose two classes
    class_1 = 1
    class_2 = 2
    # Get predicted probabilities for the two classes
    y_class_1 = y_pred == class_1
    y_class_2 = y_pred == class_2
    y_prob_1 = softmax(z)[:, class_1 - 1]
    y_prob_2 = softmax(z)[:, class_2 - 1]

    # Compute true positive rate and false positive rate for both classes
    num_thresholds = 100
    tpr_class_1 = np.zeros(num_thresholds)
    fpr_class_1 = np.zeros(num_thresholds)
    tpr_class_2 = np.zeros(num_thresholds)
    fpr_class_2 = np.zeros(num_thresholds)

    for i in range(num_thresholds):
        threshold = i / (num_thresholds - 1)
        tp_class_1 = np.sum((y_prob_1 >= threshold) & (y_class_1 == True))
        fn_class_1 = np.sum((y_prob_1 < threshold) & (y_class_1 == True))
        tn_class_1 = np.sum((y_prob_2 < threshold) & (y_class_2 == True))
        fp_class_1 = np.sum((y_prob_2 >= threshold) & (y_class_2 == False))
        tpr_class_1[i] = tp_class_1 / (tp_class_1 + fn_class_1 + 1e-8)
        fpr_class_1[i] = fp_class_1 / (fp_class_1 + tn_class_1 + 1e-8)

        tp_class_2 = np.sum((y_prob_2 >= threshold) & (y_class_2 == True))
        fn_class_2 = np.sum((y_prob_2 < threshold) & (y_class_2 == True))
        tn_class_2 = np.sum((y_prob_1 < threshold) & (y_class_1 == True))
        fp_class_2 = np.sum((y_prob_1 >= threshold) & (y_class_1 == False))
        tpr_class_2[i] = tp_class_2 / (tp_class_2 + fn_class_2 + 1e-8)
        fpr_class_2[i] = fp_class_2 / (fp_class_2 + tn_class_2 + 1e-8)

    # Plot RoC curves and confusion matrix
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax[0, 0].matshow(confusion_matrix, cmap='GnBu')
    ax[0, 0].set_xlabel("Predicted")
    ax[0, 0].set_ylabel("Actual")
    ax[0, 0].set_title("Confusion Matrix")
    for (x, y), value in np.ndenumerate(confusion_matrix):
        ax[0, 0].text(x, y, f"{value: .0f}", va="center", ha="center")

    ax[0, 1].plot(fpr_class_1, tpr_class_1, marker='x')
    ax[0, 1].set_xlabel("False positive rate")
    ax[0, 1].set_ylabel("True positive rate")                     
    ax[0, 1].set_title("ROC curve for class {}".format(class_1))

    ax[1, 0].plot(fpr_class_2, tpr_class_2, marker='x')
    ax[1, 0].set_xlabel("False positive rate")
    ax[1, 0].set_ylabel("True Positive rate")
    ax[1, 0].set_title("ROC curve for class {}".format(class_2))

    ax[1, 1].plot(fpr_class_1, tpr_class_1, marker='x', label="Class {}".format(class_1))
    ax[1, 1].plot(fpr_class_2, tpr_class_2, marker='o', label="Class {}".format(class_2))
    ax[1, 1].set_xlabel("False positive rate")
    ax[1, 1].set_ylabel("True positive rate")
    ax[1, 1].set_title("ROC curve for classes {} and {}".format(class_1, class_2))
    ax[1, 1].legend()

    fig.tight_layout()
    plt.show()


# ### Test split -- 20:80

# In[ ]:


logisticRegressor(p4["splitData"][0])


# ### Test split -- 30:70

# In[ ]:


logisticRegressor(p4["splitData"][1])


# ### Test split -- 50:50

# In[ ]:


logisticRegressor(p4["splitData"][2])


# ### Test split -- 70:30

# In[ ]:


logisticRegressor(p4["splitData"][3])


# ### Test split -- 90:10

# In[ ]:


logisticRegressor(p4["splitData"][4])


# # P7 (Neural Networks, MLP)
# 
# Train an MLP on the PCA counterpart of the KMINST dataset and report your observations
# 
# **DATA:** `p5[data]`

# ## Data handling

# In[ ]:


def stats(label, data, stats=False):
    X = data
    Y = label * np.ones(data.shape[0])
    if stats:
        return X, Y, { "mean": np.mean(X, axis=0), "cov": np.cov(X.T), "prior": data.shape[0], "data": X }
    return X, Y

classWiseData = [[] for _ in range(10)]
for row in p5["data"]:
    label = int(row[0])
    classWiseData[label].append(row[1:])
    
p5["splitData"] = [trainTestSplit(np.array(classWiseData), r, stats) for r in [0.2, 0.3, 0.5, 0.7, 0.9]]


# ## Naive Bayes

# In[ ]:


p5["result"] = [[] for _ in range(5)]


# ### Test split -- 20:80

# In[ ]:


p5["result"][0] = metrics.print(*p5["splitData"][0], naiveLogNormal)


# ### Test split -- 30:70

# In[ ]:


p5["result"][0] = metrics.print(*p5["splitData"][1], naiveLogNormal)


# ### Test split -- 50:50

# In[ ]:


p5["result"][0] = metrics.print(*p5["splitData"][2], naiveLogNormal)


# ### Test split -- 70:30

# In[ ]:


p5["result"][0] = metrics.print(*p5["splitData"][3], naiveLogNormal)


# ### Test split -- 90:10

# In[ ]:


p5["result"][0] = metrics.print(*p5["splitData"][4], naiveLogNormal)


# ## GMM

# ### Test split -- 20:80

# In[ ]:


printGmm(p5["splitData"][0])


# ### Test split -- 30:70

# In[ ]:


printGmm(p5["splitData"][1])


# ### Test split -- 50:50

# In[ ]:


printGmm(p5["splitData"][2])


# ### Test split -- 70:30

# In[ ]:


printGmm(p5["splitData"][3])


# ### Test split -- 90:10

# In[ ]:


printGmm(p5["splitData"][4])


# ## Logistic Regression

# In[ ]:


def logisticRegressor(data):
    X_train,y_train_orig , X_test, y_test_orig, classStats = data
    num_classes = 10
    num_samples = y_train_orig.shape[0]
    y_train = np.zeros((num_samples, num_classes))
    for i in range(num_samples):
        y_train[i, int(y_train_orig[i]) - 1] = 1

    # Define sigmoid function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Define softmax function
    def softmax(x):
        # subtract the maximum value from x to avoid overflow
        x -= np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        # divide by the sum of the exponential values along axis 1
        return exp_x / np.sum(exp_x, axis=1, keepdims=True, where=np.isfinite(exp_x))

    # Initialize weights and biases
    num_features = X_train.shape[1]
    W = np.random.randn(num_features, num_classes)
    b = np.random.randn(num_classes)

    # Set hyperparameters
    learning_rate = 0.1
    num_iterations = 1000
    epsilon = 1e-8

    # Train model using gradient descent
    prev_loss = float('inf')
    for i in range(num_iterations):
        # Forward propagation
        z = np.dot(X_train, W) + b
        y_pred = softmax(z)

        # Compute loss
        loss = -np.sum(y_train * np.log(y_pred + epsilon)) / num_samples

        # Backward propagation
        dz = y_pred - y_train
        dW = np.dot(X_train.T, dz) / num_samples
        db = np.sum(dz, axis=0) / num_samples

        # Update weights and biases
        W -= learning_rate * dW
        b -= learning_rate * db

        # Check stopping criterion
        if prev_loss - loss < epsilon:
            print('Stopping criterion met')
            break

        prev_loss = loss

    # Evaluate model on test set
    z = np.dot(X_test, W) + b
    y_pred = np.argmax(softmax(z), axis=1) + 1
    accuracy = np.sum(y_pred == y_test_orig) / y_test_orig.shape[0]
    print('Test accuracy:', accuracy)

    z_train = np.dot(X_train, W) + b
    y_train_pred = np.argmax(softmax(z_train), axis=1) + 1
    train_loss = -np.sum(y_train * np.log(softmax(z_train) + epsilon)) / num_samples
    train_error_rate = 1 - np.sum(y_train_pred == y_train_orig) / y_train_orig.shape[0]
    print('Training empirical risk:', train_loss)
    print('Training error rate:', train_error_rate)

    # Compute empirical risk on test data
    num_samples_test = y_test_orig.shape[0]
    y_test = np.zeros((num_samples_test, num_classes))
    for i in range(num_samples_test):
        y_test[i, int(y_test_orig[i]) - 1] = 1

    z_test = np.dot(X_test, W) + b
    test_loss = -np.sum(y_test * np.log(softmax(z_test) + epsilon)) / num_samples_test
    test_error_rate = 1 - np.sum(y_pred == y_test_orig) / y_test_orig.shape[0]
    print('Test empirical risk:', test_loss)
    print('Test error rate:', test_error_rate)

    num_classes = len(np.unique(y_test_orig))
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_test_orig)):
        true_class = int(y_test_orig[i] - 1)
        predicted_class = int(y_pred[i] - 1)
        confusion_matrix[true_class, predicted_class] += 1
    print('Confusion matrix:')
    print(confusion_matrix)

    num_classes = len(np.unique(y_test_orig))
    f1_scores = np.zeros(num_classes)
    for i in range(num_classes):
        true_positives = confusion_matrix[i, i]
        false_positives = np.sum(confusion_matrix[:, i]) - true_positives
        false_negatives = np.sum(confusion_matrix[i, :]) - true_positives
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1_scores[i] = 2 * precision * recall / (precision + recall + 1e-8)
    print('Class-wise F1 score:')
    print(f1_scores)





 # Choose two classes
    class_1 = 1
    class_2 = 2

    # Get predicted probabilities for the two classes
    y_class_1 = y_test_orig == class_1
    y_class_2 = y_test_orig == class_2
    y_prob_1 = softmax(z_test)[:, class_1 - 1]
    y_prob_2 = softmax(z_test)[:, class_2 - 1]

    # Compute true positive rate and false positive rate for both classes
    num_thresholds = 100
    tpr_class_1 = np.zeros(num_thresholds)
    fpr_class_1 = np.zeros(num_thresholds)
    tpr_class_2 = np.zeros(num_thresholds)
    fpr_class_2 = np.zeros(num_thresholds)

    for i in range(num_thresholds):
        threshold = i / (num_thresholds - 1)
        tp_class_1 = np.sum(y_class_1 & (y_prob_1 > threshold))
        fp_class_1 = np.sum(~y_class_1 & (y_prob_1 > threshold))
        tn_class_1 = np.sum(~y_class_1 & (y_prob_1 <= threshold))
        fn_class_1 = np.sum(y_class_1 & (y_prob_1 <= threshold))
        tpr_class_1[i] = tp_class_1 / (tp_class_1 + fn_class_1)
        fpr_class_1[i] = fp_class_1 / (fp_class_1 + tn_class_1)

        tp_class_2 = np.sum(y_class_2 & (y_prob_2 > threshold))
        fp_class_2 = np.sum(~y_class_2 & (y_prob_2 > threshold))
        tn_class_2 = np.sum(~y_class_2 & (y_prob_2 <= threshold))
        fn_class_2 = np.sum(y_class_2 & (y_prob_2 <= threshold))
        tpr_class_2[i] = tp_class_2 / (tp_class_2 + fn_class_2)
        fpr_class_2[i] = fp_class_2 / (fp_class_2 + tn_class_2)
        
    
    # Plot RoC curves and confusion matrix
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax[0, 0].matshow(confusion_matrix, cmap='GnBu')
    ax[0, 0].set_xlabel("Predicted")
    ax[0, 0].set_ylabel("Actual")
    ax[0, 0].set_title("Confusion Matrix")
    for (x, y), value in np.ndenumerate(confusion_matrix):
        ax[0, 0].text(x, y, f"{value: .0f}", va="center", ha="center")


    ax[0, 1].plot(fpr_class_1, tpr_class_1, marker='x')
    ax[0, 1].set_xlabel("False positive rate")
    ax[0, 1].set_ylabel("True positive rate")                     
    ax[0, 1].set_title("ROC curve for class {}".format(class_1))

    ax[1, 0].plot(fpr_class_2, tpr_class_2, marker='x')
    ax[1, 0].set_xlabel("False positive rate")
    ax[1, 0].set_ylabel("True Positive rate")
    ax[1, 0].set_title("ROC curve for class {}".format(class_2))

    ax[1, 1].plot(fpr_class_1, tpr_class_1, marker='x', label="Class {}".format(class_1))
    ax[1, 1].plot(fpr_class_2, tpr_class_2, marker='o', label="Class {}".format(class_2))
    ax[1, 1].set_xlabel("False positive rate")
    ax[1, 1].set_ylabel("True positive rate")
    ax[1, 1].set_title("ROC curve for classes {} and {}".format(class_1, class_2))
    ax[1, 1].legend()

    fig.tight_layout()
    plt.show()




# ### Test split -- 20:80

# In[ ]:


logisticRegressor(p5["splitData"][0])


# ### Test split -- 30:70

# In[ ]:


logisticRegressor(p5["splitData"][1])


# ### Test split -- 50:50

# In[ ]:


logisticRegressor(p5["splitData"][2])


# ### Test split -- 70:30

# In[ ]:


logisticRegressor(p5["splitData"][3])


# ### Test split -- 90:10

# In[ ]:


logisticRegressor(p5["splitData"][4])


# In[ ]:




