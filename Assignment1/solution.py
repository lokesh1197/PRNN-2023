#!/usr/bin/env python
# coding: utf-8

# # Initialization

# In[135]:


import numpy as np
from matplotlib import pyplot as plt

# for p-value out of significance test
from scipy.stats import ttest_ind


# In[136]:


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

# In[137]:


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

# In[138]:


p1["train"].shape


# In[139]:


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

# In[140]:


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

# In[141]:


p2["train"].shape, p2["test"].shape


# In[142]:


X, Y = parseData(p2["train"])
X_test, Y_test = parseData(p2["test"])

# Initialise the parameters to be a null vector
W = np.array([0, 0, 0, 0])

# Check metrics with parameters as null vector
print(mse(X, Y, W))
print(mae(X, Y, W))


# ## Linear regression
# $h(x) = w_0 + w_1x_1 + w_2x_2 + w_3x_3$

# In[143]:


p2["result"] = [[] for _ in range(8)]
p2["result"][0] = printResult(X, Y, X_test, Y_test)


# ## Non-Linear regression (1)
# $h_1(x) = w_0 + w_1x_1 + w_2x_2 + w_3x_3 + w_4x_1x_2 + w_5x_2x_3 + w_6x_3x_1 + w_7x_1^2 + w_8x_2^2 + w_9x_3^2$

# In[144]:


def makeQuadratic(data):
    n = data.shape[1]
    return np.array([data.T[i] * data.T[j] for i in range(n) for j in range(n) if j <= i]).T

p2["result"][1] = printResult(X, Y, X_test, Y_test, makeQuadratic)


# ## Non-Linear regression (2)
# $h_2(x) = w_0 + w_1x_1 + w_2x_2 + w_3x_3$
#      $+ w_4x_1x_2 + w_5x_2x_3 + w_6x_3x_1 + w_7x_1^2 + w_8x_2^2 + w_9x_3^2$
#      $+ w_{10}x_1x_2x_3 + w_{11}x_1^2x_2 + w_{12}x_2^2x_1 + w_{13}x_2^2x_3 + w_{14}x_3^2x_2 + w_{15}x_3^2x_1 + w_{16}x_1^2x_3 + w_{17}x_1^3 + w_{18}x_2^3 + w_{19}x_3^3$

# In[145]:


def makeCubic(data):
    n = data.shape[1]
    return np.array([data.T[i] * data.T[j] * data.T[k] for i in range(n) for j in range(n) for k in range(n) if j <= i and k <= j]).T

p2["result"][2] = printResult(X, Y, X_test, Y_test, makeCubic)


# ## Non-Linear regression (3)
# $h_3(x) = h_1(h_1(x))$

# In[146]:


p2["result"][3] = printResult(X, Y, X_test, Y_test, lambda x : makeQuadratic(makeQuadratic(x)))


# ## Non-Linear regression (4)
# $h_4(x) = h_1(h_2(x))$

# In[147]:


p2["result"][4] = printResult(X, Y, X_test, Y_test, lambda x : makeQuadratic(makeCubic(x)))


# ## Non-Linear regression (5)
# $h_5(x) = h_2(h_2(x))$

# In[148]:


p2["result"][5] = printResult(X, Y, X_test, Y_test, lambda x : makeCubic(makeCubic(x)))


# ## Non-Linear regression (6)
# $\phi(x) = \sum_{i=0, j=0, i \neq j}^n w_{ij} x_ie^{x_j} + \sum_{i=0}^n w_{ii} x_ie^{x_i}$
# 
# $h_6(x) = \phi(h_1(x))$

# In[149]:


def makeExp(data):
    n = data.shape[1]
    return np.array([data.T[i] * np.exp(data.T[j]) for i in range(n) for j in range(n) if i != j] + [data.T[i] * np.exp(data.T[i]) for i in range(n)]).T

p2["result"][6] = printResult(X, Y, X_test, Y_test, lambda x : makeExp(makeQuadratic(x)))


# ## Non-Linear regression (7)
# $h_7(x) = \phi(h_2(x))$

# In[150]:


p2["result"][7] = printResult(X, Y, X_test, Y_test, lambda x : makeExp(makeCubic(x)))


# In[151]:


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


# In[152]:


a = np.array([[1, 2,2], [4, 3,5]])
(a.T / np.sum(a, axis=1)).T


# In[ ]:





# In[153]:


# means = [np.mean(a) + np.random.rand(3)]
covs = [np.diag(np.random.rand(3) * 2 + 0.5)]
print(covs)


# # Custom functions for P3, P4 and P5

# In[289]:


np.exp(-np.log(np.finfo("float64").max)) / 1000


# In[327]:


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

# X -> data, k -> number of normal densities
def em(X, k, max_iter = 100, random_seed=42):
    # m -> number of datapoints, n -> number of features
    m, n = X.shape
    eps = 1e-8
    # initialization
    np.random.seed(random_seed)
    weights = np.random.dirichlet(np.ones(k))
    means = [np.mean(X) + np.random.rand(n) for _ in range(k)]
    covs = [np.diag(np.random.rand(n) * 2 + 0.5) for _ in range(k)]
    
    for i in range(max_iter):
        # if np.sum(weights) == 0:
        #     weights = np.ones(k) / k
            
        # Expectation step
        probs = np.array([[normal(x, means[i], covs[i]) * weights[i] for i in range(k)] for x in X])
        probs = (probs.T / np.sum(probs, axis=1)).T
        
        # Maximization step
        old_means = means.copy()
        covs = [(X - means[i]).T @ np.diag(probs.T[i]) @ (X - means[i]) / (np.sum(probs.T[i]) + 1e-8) for i in range(k)]
        means = [X.T @ probs.T[i] / (np.sum(probs.T[i]) + 1e-8)  for i in range(k)]
        weights = np.sum(probs, axis=0) / m
        
        if np.linalg.norm(np.array(means) - np.array(old_means)) < 1e-8:
            break
        
    return weights, means, covs

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


# In[317]:


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

# ## Data Handling

# In[156]:


p3["train"].shape, p3["test"].shape


# In[157]:


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


# In[320]:


def splitData(data):
    # X = np.array([normalize(col) for col in data.T[:-1]]).T
    X = data.T[:-1].T
    Y = data.T[-1].T.astype("int") - 1
    return X, Y

X, Y = splitData(p3["train"])
X_test, Y_test = splitData(p3["test"])

X.shape, Y.shape, X_test.shape, Y_test.shape


# ## Bayes' classifier with normal distribution

# In[318]:


p3["result"] = [[] for _ in range(5)]
p3["result"][0] = metrics.print(X, Y, X_test, Y_test, classStats, logNormal)


# ## Bayes' classifier with exponential distribution

# In[160]:


p3["result"][1] = metrics.print(X, Y, X_test, Y_test, classStats, logExp)


# ## Bayes' classifier with GMM distribution

# In[299]:


def printGmmP3(number_of_guassians , max_iter = 50):
    classStatsGMM = {}
    for label in classStats:
        classStatsGMM[label] = { "prior": classStats[label]["prior"] }
        classStatsGMM[label]["weights"], classStatsGMM[label]["mean"], classStatsGMM[label]["cov"] = em(classStats[label]["data"], number_of_guassians, max_iter)
        print("weights of class ", str(label + 1), ": ", classStatsGMM[label]["weights"])

    metrics.print(X, Y, X_test, Y_test, classStatsGMM, logGMM, result=False)


# In[300]:


printGmmP3(2)


# In[301]:


printGmmP3(5)


# In[ ]:


printGmmP3(8)


# ## Logistic Regression

# In[218]:


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

# In[232]:


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

# ## Data handling

# In[245]:


get_ipython().run_cell_magic('capture', '', '!unzip -n data/images.zip -d data')


# In[246]:


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


# In[247]:


fig, ax = plt.subplots(2, 5, figsize=(12, 4))
for i in range(p4["data"].shape[0]):
    ax[i // 5][i % 5].imshow(p4["data"][i][0].astype(np.uint8), cmap='gray')
    ax[i // 5][i % 5].set_title(str(i))
    ax[i // 5][i % 5].get_xaxis().set_visible(False)
    ax[i // 5][i % 5].get_yaxis().set_visible(False)

fig.tight_layout()


# In[334]:


def imgToFeatures(label, data, stats=False):
    X = np.array([x.flatten() for x in data]) / 255
    Y = label * np.ones(data.shape[0])
    if stats:
        return X, Y, { "mean": np.mean(X, axis=0), "cov": np.cov(X.T), "prior": data.shape[0], "data": X }
    return X, Y

p4["splitData"] = [trainTestSplit(p4["data"], r, imgToFeatures) for r in [0.2, 0.3, 0.5, 0.7, 0.9]]


# ## Naive Bayes

# In[249]:


p4["result"] = [[] for _ in range(5)]


# ### Test split -- 20:80

# In[250]:


p4["result"][0] = metrics.print(*p4["splitData"][0], naiveLogNormal)


# ### Test split -- 30:70

# In[251]:


p4["result"][0] = metrics.print(*p4["splitData"][1], naiveLogNormal)


# ### Test split -- 50:50

# In[252]:


p4["result"][0] = metrics.print(*p4["splitData"][2], naiveLogNormal)


# ### Test split -- 70:30

# In[253]:


p4["result"][0] = metrics.print(*p4["splitData"][3], naiveLogNormal)


# ### Test split -- 90:10

# In[254]:


p4["result"][0] = metrics.print(*p4["splitData"][4], naiveLogNormal)


# ## GMM

# In[335]:


def printGmm(data, number_of_guassians=2):
    classStatsGMM = {}
    for label in data[-1]:
        classStatsGMM[label] = { "prior": data[-1][label]["prior"] }
        classStatsGMM[label]["weights"], classStatsGMM[label]["mean"], classStatsGMM[label]["cov"] = em(data[-1][label]["data"], number_of_guassians, 50)

    metrics.print(*data[:-1], classStatsGMM, logGMM, result=False)


# ### Test split -- 20:80

# In[336]:


printGmm(p4["splitData"][0])


# ### Test split -- 30:70

# In[ ]:


printGmm(p4["splitData"][1])


# ### Test split -- 50:50

# In[342]:


printGmm(p4["splitData"][2])


# ### Test split -- 70:30

# In[ ]:


printGmm(p4["splitData"][3])


# ### Test split -- 90:10

# In[ ]:


printGmm(p4["splitData"][4])


# ## Logistic Regression

# In[223]:


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

# In[224]:


logisticRegressor(p4["splitData"][0])


# ### Test split -- 30:70

# In[225]:


logisticRegressor(p4["splitData"][1])


# ### Test split -- 50:50

# In[227]:


logisticRegressor(p4["splitData"][2])


# ### Test split -- 70:30

# In[ ]:


logisticRegressor(p4["splitData"][3])


# ### Test split -- 90:10

# In[228]:


logisticRegressor(p4["splitData"][4])


# # P5 (Multi-class classification)
# 
# In this part, the data from the previous problem is ’condensed’ (using PCA) to **10 dimensions**. Repeat the above experiment with all the models and metrics and record your observations.
# 
# **DATA:** `PCA_MNIST`(KannadaMNISTPCA.csv)

# ## Data handling

# In[256]:


p5["data"] = np.genfromtxt(dataFolder + "/PCA_MNIST.csv", delimiter=',')[1:]


# In[204]:


p5["data"].shape


# In[337]:


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

# In[257]:


p5["result"] = [[] for _ in range(5)]


# ### Test split -- 20:80

# In[258]:


p5["result"][0] = metrics.print(*p5["splitData"][0], naiveLogNormal)


# ### Test split -- 30:70

# In[ ]:


p5["result"][0] = metrics.print(*p5["splitData"][1], naiveLogNormal)


# ### Test split -- 50:50

# In[259]:


p5["result"][0] = metrics.print(*p5["splitData"][2], naiveLogNormal)


# ### Test split -- 70:30

# In[ ]:


p5["result"][0] = metrics.print(*p5["splitData"][3], naiveLogNormal)


# ### Test split -- 90:10

# In[260]:


p5["result"][0] = metrics.print(*p5["splitData"][4], naiveLogNormal)


# ## GMM

# ### Test split -- 20:80

# In[338]:


printGmm(p5["splitData"][0])


# ### Test split -- 30:70

# In[ ]:


printGmm(p5["splitData"][1])


# ### Test split -- 50:50

# In[340]:


printGmm(p5["splitData"][2])


# ### Test split -- 70:30

# In[ ]:


printGmm(p5["splitData"][3])


# ### Test split -- 90:10

# In[341]:


printGmm(p5["splitData"][4])


# ## Logistic Regression

# In[270]:


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

# In[271]:


logisticRegressor(p5["splitData"][0])


# ### Test split -- 30:70

# In[272]:


logisticRegressor(p5["splitData"][1])


# ### Test split -- 50:50

# In[273]:


logisticRegressor(p5["splitData"][2])


# ### Test split -- 70:30

# In[274]:


logisticRegressor(p5["splitData"][3])


# ### Test split -- 90:10

# In[275]:


logisticRegressor(p5["splitData"][4])


# In[ ]:




