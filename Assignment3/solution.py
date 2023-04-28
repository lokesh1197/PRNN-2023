#!/usr/bin/env python
# coding: utf-8

# # Initialization

# In[6]:


import numpy as np
from matplotlib import pyplot as plt

# for image data handling
import os
from os.path import join, isfile, dirname
from PIL import Image


# # Data Handling

# ### Uncompress compressed files

# In[7]:


get_ipython().run_cell_magic('capture', '', '!unzip -n ../data/images.zip -d ../data')


# ### Custom functions

# In[8]:


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
    n = len(data)
    m = int(np.floor(data.shape[1] * train_ratio))
    classStats = {}
    x_train, y_train, x_test, y_test = [[[] for _ in range(n)] for _ in range(4)]
    for label in range(n):
        x_train[label], y_train[label], classStats[label] = func(label, data[label][:m], True)
        x_test[label], y_test[label] = func(label, data[label][m:])
    
    X, Y, X_test, Y_test = [x.reshape(-1, x.shape[-1]) for x in [np.array(x) for x in [x_train, y_train, x_test, y_test]]]
    return X, np.array(Y.flatten(), dtype=int), X_test, np.array(Y_test.flatten(), dtype=int), classStats

def imgToFeatures(label, data, stats=False):
    X = np.array([x.flatten() for x in data]) / 255
    Y = label * np.ones(data.shape[0])
    if stats:
        return X, Y, { "mean": np.mean(X, axis=0), "cov": np.cov(X.T), "prior": data.shape[0], "data": X }
    return X, Y

def stats(label, data, stats=False):
    X = data
    Y = label * np.ones(data.shape[0])
    if stats:
        return X, Y, { "mean": np.mean(X, axis=0), "cov": np.cov(X.T), "prior": data.shape[0], "data": X }
    return X, Y


# ### Data extraction

# In[9]:


dataFolder = "../data"
imageDir = join(dataFolder, "images")

# p1 = { "testDir": dataFolder + "/p1_test.csv", "trainDir": dataFolder + "/p1_train.csv" } # regression
# p2 = { "testDir": dataFolder + "/p2_test.csv", "trainDir": dataFolder + "/p2_train.csv" } # regression
p3 = { "testDir": dataFolder + "/p3_test.csv", "trainDir": dataFolder + "/p3_train.csv" } # classification
p4 = {}                                                                                   # classification
p5 = {}                                                                                   # classification

# p1["test"] = np.genfromtxt(p1["testDir"], delimiter=',')
# p1["train"] = np.genfromtxt(p1["trainDir"], delimiter=',')
# p2["test"] = np.genfromtxt(p2["testDir"], delimiter=',')
# p2["train"] = np.genfromtxt(p2["trainDir"], delimiter=',')
p3["test"] = np.genfromtxt(p3["testDir"], delimiter=',')
p3["train"] = np.genfromtxt(p3["trainDir"], delimiter=',')
p4["data"] = genFromImage(imageDir)
p5["data"] = np.genfromtxt(dataFolder + "/PCA_MNIST.csv", delimiter=',')[1:]

print("--------------------------- Data Shapes ------------------------------")
# print("    (Regression) p1[train]:      ", p1["train"].shape, ", p1[test]: ", p1["test"].shape)
# print("    (Regression) p2[train]:      ", p2["train"].shape, ", p2[test]: ", p2["test"].shape)
print("(Classification) p3[train]:     ", p3["train"].shape, ", p3[test]: ", p3["test"].shape)
print("(Classification)  p4[data]:", p4["data"].shape)
print("(Classification)  p5[data]:     ", p5["data"].shape)


# In[10]:


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


# In[11]:


p4["X"], p4["Y"], p4["X_test"], p4["Y_test"], p4["classStats"] = trainTestSplit(p4["data"], 0.7, imgToFeatures)

p4["X"].shape, p4["Y"].shape, p4["X_test"].shape, p4["Y_test"].shape


# In[12]:


classWiseData = [[] for _ in range(10)]
for row in p5["data"]:
    label = int(row[0])
    classWiseData[label].append(row[1:])
    
p5["X"], p5["Y"], p5["X_test"], p5["Y_test"], p5["classStats"] = trainTestSplit(np.array(classWiseData), 0.5, stats)
p5["X"].shape, p5["Y"].shape, p5["X_test"].shape, p5["Y_test"].shape


# In[13]:


fig, ax = plt.subplots(2, 5, figsize=(12, 4))
for i in range(p4["data"].shape[0]):
    ax[i // 5][i % 5].imshow(p4["data"][i][0].astype(np.uint8), cmap='gray')
    ax[i // 5][i % 5].set_title(str(i))
    ax[i // 5][i % 5].get_xaxis().set_visible(False)
    ax[i // 5][i % 5].get_yaxis().set_visible(False)

fig.tight_layout()


# # Metrics

# In[14]:


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
        p = np.zeros(cnf.shape[1])
        r = np.zeros(cnf.shape[1])
        f1 = np.zeros(cnf.shape[1])
        for i in range(f1.size):
            TP = cnf[i][i]
            FP, FN = sum_predict[i] - TP, sum_actual[i] - TP
            p[i], r[i] = TP/(TP + FP + 1e-8), TP/(TP + FN + 1e-8)
            f1[i] = 2 * p[i] * r[i] / (p[i] + r[i] + 1e-8)
        return f1, p, r 

    def printCnf(cnf_train, cnf_test):
        print("Confusion Matrix:")
        print(cnf_test)
        
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].matshow(cnf_train.T, cmap='GnBu')
        ax[0].set_xlabel("Predicted")
        ax[0].set_ylabel("Actual")
        ax[0].set_title("Confusion Matrix (train)")
        for (x, y), value in np.ndenumerate(cnf_train):
            ax[0].text(x, y, f"{value: .0f}", va="center", ha="center")
        
        ax[1].matshow(cnf_test.T, cmap='GnBu')
        ax[1].set_xlabel("Predicted")
        ax[1].set_ylabel("Actual")
        ax[1].set_title("Confusion Matrix (test)")
        for (x, y), value in np.ndenumerate(cnf_test):
            ax[1].text(x, y, f"{value: .0f}", va="center", ha="center")

    
    def print(pred, Y, pred_test, Y_test, visualize=True, result=False):
        n_labels = len(np.unique(Y))
                
        cnf_train = metrics.confusionMatrix(pred, Y, n_labels)
        cnf_test = metrics.confusionMatrix(pred_test, Y_test, n_labels)
        acc_train = metrics.accuracy(pred, Y)
        acc_test = metrics.accuracy(pred_test, Y_test)
        f1_train, p_train, r_train = metrics.f1Score(cnf_train)
        f1_test, p_test, r_test = metrics.f1Score(cnf_test)
        
        if visualize:
            print("------------------ Train ---------------------")
            print(f"Classification Accuracy : {acc_train * 100:.2f}%")
            print(f"Average F1 Score        : {np.average(f1_train):.2f}")
            print(f"Average Precision       : {np.average(p_train):.2f}")
            print(f"Average Recall          : {np.average(r_train):.2f}")
            print("------------------ Test ----------------------")
            print(f"Classification Accuracy : {acc_test * 100:.2f}%")
            print(f"Average F1 Score        : {np.average(f1_test):.2f}")
            print(f"Average Precision       : {np.average(p_test):.2f}")
            print(f"Average Recall          : {np.average(r_test):.2f}")
            
            metrics.printCnf(cnf_train, cnf_test)
        
        if result:
            return [acc_train, f1_train, p_train, r_train, cnf_train], [acc_test, f1_test, p_test, r_test, cnf_test]


# # Problem 1
# 
# Implement classification trees using Gini impurity and cross-entropy as impurity functions with different depths. 
# 
# For the MNIST problem, consider the PCA data
# 
# Data: `p3, p5`

# ## Implementation

# ### Impurity Funcions

# In[15]:


def gini(data):
    n = data.shape[0]
    if n == 0:
        return 0
    p = np.bincount(data) / n
    return 1 - np.sum(p ** 2)

print("Gini (test): ", gini(p3["Y"]), gini(p3['Y'][p3["Y"] == 0]))

def entropy(data):
    n = data.shape[0]
    if n == 0:
        return 0
    p = np.bincount(data) / n
    return -np.sum(p * np.log2(p + 1e-8))

print("Entropy (test): ", entropy(p3["Y"]), entropy(p3['Y'][p3["Y"] == 0]))


# ### CART algorithm

# In[16]:


# restricting the number of thresholds to increase the speed of the algorithm
def recommendedThresholds(col, y):
    a = np.c_[col, y]
    a.sort(axis=0)
    b = []
    for i in range(1, a.shape[0]):
        if a[i][1] != a[i - 1][1]:
            b.append((a[i][0] + a[i - 1][0]) / 2)
    return b

def recommendedSplit(X, y, impurity, skip_features_idx=[]):
    best_impurity = 1e9
    best_split = None
    for i, col in enumerate(X.T):
        if i in skip_features_idx:
            continue
        ts = recommendedThresholds(col, y)
        for t in ts:
            left, right = y[col <= t], y[col > t]
            impurity_split = (impurity(left) * left.size + impurity(right) * right.size) / y.size
            if impurity_split < best_impurity:
                best_impurity = impurity_split
                best_split = (i, t)
    return best_split

recommendedSplit(p3["X"], p3["Y"], gini)


# ### Decision Tree

# In[17]:


# inner nodes are [i, t, left, right]
# leaf node is a number (class label)
# returns a tree in pre-order traversal format
def buildTree(X, y, impurity, max_depth, depth=0, skip_features_idx=[]):
    if depth == max_depth:
        return [np.argmax(np.bincount(y))]
    if np.unique(y).size == 1:
        return [y[0]]
    
    split = recommendedSplit(X, y, impurity, skip_features_idx)

    if split is None:
        return [np.argmax(np.bincount(y))]
    
    i, t = split
    left, right = y[X[:, i] <= t], y[X[:, i] > t]
    leftX, rightX = X[X[:, i] <= t], X[X[:, i] > t]

    if left.size == 0 or right.size == 0:
        return [np.argmax(np.bincount(y))]
    
    return [split, buildTree(leftX, left, impurity, max_depth, depth + 1), buildTree(rightX, right, impurity, max_depth, depth + 1)]

def printTree(tree, depth=0):
    if len(tree) == 1:
        print(" " * depth, "Leaf: ", tree[0])
    else:
        print(" " * depth, f"Split: {tree[0][0]:.2f} <=  {tree[0][1]:.2f}")
        printTree(tree[1], depth + 1)
        printTree(tree[2], depth + 1)

def predict(X, tree):
    if len(tree) == 1:
        return tree[0]
    i, t = tree[0]
    return predict(X, tree[1]) if X[i] <= t else predict(X, tree[2])
    
def predictAll(X, tree):
    return np.array([predict(x, tree) for x in X])


# ## Experiment on P3 Data

# ### Using Gini as impurity measure

# In[47]:


depths = [3, 5, 7, 9, 11, 13, 17, 20]
results = []

for depth in depths:
    tree = buildTree(p3["X"], p3["Y"], gini, depth)
    results.append(metrics.print(predictAll(p3["X"], tree), p3["Y"], predictAll(p3["X_test"], tree), p3["Y_test"], visualize=False, result=True))

bestResult_p3_gini = results[np.argmax([row[1][0] for row in results])]

plt.plot(depths, [row[0][0] for row in results], label="train", marker='o')
plt.plot(depths, [row[1][0] for row in results], label="test", marker='x')
plt.xlabel("Complexity (Depth) -->")
plt.ylabel("Accuracy -->")
plt.legend()
plt.title("Accuracy vs complexity")

plt.show()


# In[48]:


print(f"Test Accuracy: {bestResult_p3_gini[1][0] * 100:.2f}%, Precision: {np.average(bestResult_p3_gini[1][2]):.2f}, Recall: {np.average(bestResult_p3_gini[1][3]):.2f}, F1 Score: {np.average(bestResult_p3_gini[1][1]):.2f}")
metrics.printCnf(bestResult_p3_gini[0][4], bestResult_p3_gini[1][4])


# ### Using Cross-Entropy as impurity measure

# In[49]:


depths = [3, 5, 7, 9, 11, 13, 17, 20]
results = []

for depth in depths:
    tree = buildTree(p3["X"], p3["Y"], entropy, depth)
    results.append(metrics.print(predictAll(p3["X"], tree), p3["Y"], predictAll(p3["X_test"], tree), p3["Y_test"], visualize=False, result=True))

bestResult_p3_entropy = results[np.argmax([row[1][0] for row in results])]

plt.plot(depths, [row[0][0] for row in results], label="train", marker='o')
plt.plot(depths, [row[1][0] for row in results], label="test", marker='x')
plt.xlabel("Complexity (Depth) -->")
plt.ylabel("Accuracy -->")
plt.legend()
plt.title("Accuracy vs complexity")

plt.show()


# In[50]:


print(f"Test Accuracy: {bestResult_p3_entropy[1][0] * 100:.2f}%, Precision: {np.average(bestResult_p3_entropy[1][2]):.2f}, Recall: {np.average(bestResult_p3_entropy[1][3]):.2f}, F1 Score: {np.average(bestResult_p3_entropy[1][1]):.2f}")
metrics.printCnf(bestResult_p3_entropy[0][4], bestResult_p3_entropy[1][4])


# ## Experiment on P5 Data (MNIST PCA)

# ### Using Gini as impurity measure

# In[51]:


depths = [3, 5, 7, 9, 11, 13, 17, 20]
results = []

for depth in depths:
    tree = buildTree(p5["X"], p5["Y"], gini, depth)
    results.append(metrics.print(predictAll(p5["X"], tree), p5["Y"], predictAll(p5["X_test"], tree), p5["Y_test"], visualize=False, result=True))

bestResult_p5_gini = results[np.argmax([row[1][0] for row in results])]

plt.plot(depths, [row[0][0] for row in results], label="train", marker='o')
plt.plot(depths, [row[1][0] for row in results], label="test", marker='x')
plt.xlabel("Complexity (Depth) -->")
plt.ylabel("Accuracy -->")
plt.legend()
plt.title("Accuracy vs complexity")

plt.show()


# In[52]:


print(f"Test Accuracy: {bestResult_p5_gini[1][0] * 100:.2f}%, Precision: {np.average(bestResult_p5_gini[1][2]):.2f}, Recall: {np.average(bestResult_p5_gini[1][3]):.2f}, F1 Score: {np.average(bestResult_p5_gini[1][1]):.2f}")
metrics.printCnf(bestResult_p5_gini[0][4], bestResult_p5_gini[1][4])


# ### Using Entropy as impurity measure

# In[53]:


depths = [3, 5, 7, 9, 11, 13, 17, 20]
results = []

for depth in depths:
    tree = buildTree(p5["X"], p5["Y"], entropy, depth)
    results.append(metrics.print(predictAll(p5["X"], tree), p5["Y"], predictAll(p5["X_test"], tree), p5["Y_test"], visualize=False, result=True))

bestResult_p5_entropy = results[np.argmax([row[1][0] for row in results])]

plt.plot(depths, [row[0][0] for row in results], label="train", marker='o')
plt.plot(depths, [row[1][0] for row in results], label="test", marker='x')
plt.xlabel("Complexity (Depth) -->")
plt.ylabel("Accuracy -->")
plt.legend()
plt.title("Accuracy vs complexity")

plt.show()


# In[54]:


print(f"Test Accuracy: {bestResult_p5_entropy[1][0] * 100:.2f}%, Precision: {np.average(bestResult_p5_entropy[1][2]):.2f}, Recall: {np.average(bestResult_p5_entropy[1][3]):.2f}, F1 Score: {np.average(bestResult_p5_entropy[1][1]):.2f}")
metrics.printCnf(bestResult_p5_entropy[0][4], bestResult_p5_entropy[1][4])


# # Problem 2
# 
# Implement Random forest Algorithm with varying numbers of trees and features and report your observations
# 
# Data: `p3, p4, p5`

# ## Implementation

# In[18]:


def bootstrap(X, Y):
    n = len(X)
    idx = np.random.choice(n, n, replace=False)
    return X[idx], Y[idx]

def random_forest(X, Y, n_trees=10, depth=5, impurity=gini, n_features=None):
    trees = []
    for _ in range(n_trees):
        X_train, Y_train = bootstrap(X, Y)
        idx = [] if n_features is None else np.random.choice(X_train.shape[1], X_train.shape[1] - n_features, replace=False)
        trees.append(buildTree(X_train, Y_train, impurity, depth, skip_features_idx=idx))
    return trees

def predictForest(X, trees):
    predictions = np.array([predictAll(X, tree) for tree in trees])
    return np.array([np.bincount(prediction).argmax() for prediction in predictions.T])

def predictForestProb(X, trees):
    predictions = np.array([predictAll(X, tree) for tree in trees])
    return np.array([np.bincount(prediction) / len(prediction) for prediction in predictions.T])


# ## Experiment on P3 data

# ### Number of trees: 10, Number of features: 10

# In[36]:


trees = random_forest(p3["X"], p3["Y"], n_trees=10, depth=15, impurity=gini)
metrics.print(predictForest(p3["X"], trees), p3["Y"], predictForest(p3["X_test"], trees), p3["Y_test"], visualize=True)


# ### Number of trees: 15, Number of features: 10

# In[37]:


trees = random_forest(p3["X"], p3["Y"], n_trees=15, depth=15, impurity=gini)
metrics.print(predictForest(p3["X"], trees), p3["Y"], predictForest(p3["X_test"], trees), p3["Y_test"], visualize=True)


# ### Number of trees: 20, Number of features: 10

# In[38]:


trees = random_forest(p3["X"], p3["Y"], n_trees=20, depth=15, impurity=gini)
metrics.print(predictForest(p3["X"], trees), p3["Y"], predictForest(p3["X_test"], trees), p3["Y_test"], visualize=True)


# ### Number of trees: 25, Number of features: 10

# In[39]:


trees = random_forest(p3["X"], p3["Y"], n_trees=25, depth=15, impurity=gini)
metrics.print(predictForest(p3["X"], trees), p3["Y"], predictForest(p3["X_test"], trees), p3["Y_test"], visualize=True)


# ### Number of trees: 20, Number of features: 7

# In[40]:


trees = random_forest(p3["X"], p3["Y"], n_trees=20, depth=15, impurity=gini, n_features=7)
metrics.print(predictForest(p3["X"], trees), p3["Y"], predictForest(p3["X_test"], trees), p3["Y_test"], visualize=True)


# ### Number of trees: 20, Number of features: 5

# In[41]:


trees = random_forest(p3["X"], p3["Y"], n_trees=20, depth=15, impurity=gini, n_features=5)
metrics.print(predictForest(p3["X"], trees), p3["Y"], predictForest(p3["X_test"], trees), p3["Y_test"], visualize=True)


# ### Number of trees: 20, Number of features: 3

# In[42]:


trees = random_forest(p3["X"], p3["Y"], n_trees=20, depth=15, impurity=gini, n_features=3)
metrics.print(predictForest(p3["X"], trees), p3["Y"], predictForest(p3["X_test"], trees), p3["Y_test"], visualize=True)


# ## Experiment on P4 data

# ### Number of trees: 10, Number of features: 64

# In[43]:


trees = random_forest(p4["X"], p4["Y"], n_trees=10, depth=15, impurity=gini)
metrics.print(predictForest(p4["X"], trees), p4["Y"], predictForest(p4["X_test"], trees), p4["Y_test"], visualize=True)


# ### Number of trees: 15, Number of features: 64

# In[44]:


trees = random_forest(p4["X"], p4["Y"], n_trees=15, depth=15, impurity=gini)
metrics.print(predictForest(p4["X"], trees), p4["Y"], predictForest(p4["X_test"], trees), p4["Y_test"], visualize=True)


# ### Number of trees: 20, Number of features: 64

# In[45]:


trees = random_forest(p4["X"], p4["Y"], n_trees=20, depth=15, impurity=gini)
metrics.print(predictForest(p4["X"], trees), p4["Y"], predictForest(p4["X_test"], trees), p4["Y_test"], visualize=True)


# ### Number of trees: 25, Number of features: 64

# In[46]:


trees = random_forest(p4["X"], p4["Y"], n_trees=25, depth=15, impurity=gini)
metrics.print(predictForest(p4["X"], trees), p4["Y"], predictForest(p4["X_test"], trees), p4["Y_test"], visualize=True)


# ### Number of trees: 20, Number of features: 16

# In[47]:


trees = random_forest(p4["X"], p4["Y"], n_trees=20, depth=15, impurity=gini, n_features=16)
metrics.print(predictForest(p4["X"], trees), p4["Y"], predictForest(p4["X_test"], trees), p4["Y_test"], visualize=True)


# ### Number of trees: 20, Number of features: 4

# In[48]:


trees = random_forest(p4["X"], p4["Y"], n_trees=20, depth=15, impurity=gini, n_features=4)
metrics.print(predictForest(p4["X"], trees), p4["Y"], predictForest(p4["X_test"], trees), p4["Y_test"], visualize=True)


# ## Experiment on P5 data

# ### Number of trees: 10, Number of features: 10

# In[49]:


trees = random_forest(p5["X"], p5["Y"], n_trees=10, depth=15, impurity=gini)
metrics.print(predictForest(p5["X"], trees), p5["Y"], predictForest(p5["X_test"], trees), p5["Y_test"], visualize=True)


# ### Number of trees: 15, Number of features: 10

# In[50]:


trees = random_forest(p5["X"], p5["Y"], n_trees=15, depth=15, impurity=gini)
metrics.print(predictForest(p5["X"], trees), p5["Y"], predictForest(p5["X_test"], trees), p5["Y_test"], visualize=True)


# ### Number of trees: 20, Number of features: 10

# In[51]:


trees = random_forest(p5["X"], p5["Y"], n_trees=20, depth=15, impurity=gini)
metrics.print(predictForest(p5["X"], trees), p5["Y"], predictForest(p5["X_test"], trees), p5["Y_test"], visualize=True)


# ### Number of trees: 25, Number of features: 10

# In[52]:


trees = random_forest(p5["X"], p5["Y"], n_trees=25, depth=15, impurity=gini)
metrics.print(predictForest(p5["X"], trees), p5["Y"], predictForest(p5["X_test"], trees), p5["Y_test"], visualize=True)


# ### Number of trees: 20, Number of features: 7

# In[53]:


trees = random_forest(p5["X"], p5["Y"], n_trees=20, depth=15, impurity=gini, n_features=7)
metrics.print(predictForest(p5["X"], trees), p5["Y"], predictForest(p5["X_test"], trees), p5["Y_test"], visualize=True)


# ### Number of trees: 20, Number of features: 5

# In[54]:


trees = random_forest(p5["X"], p5["Y"], n_trees=20, depth=15, impurity=gini, n_features=5)
metrics.print(predictForest(p5["X"], trees), p5["Y"], predictForest(p5["X_test"], trees), p5["Y_test"], visualize=True)


# ### Number of trees: 20, Number of features: 3

# In[55]:


trees = random_forest(p5["X"], p5["Y"], n_trees=20, depth=15, impurity=gini, n_features=3)
metrics.print(predictForest(p5["X"], trees), p5["Y"], predictForest(p5["X_test"], trees), p5["Y_test"], visualize=True)


# # Problem 3
# 
# Implement the Adaboost algorithm with at least 3 learners and one of them must be a Neural Network (MLP/CNN). 
# 
# Report the comparison between this and using only one classifier. 
# 
# Plot the convergence of train error as a function of the number of learners.
# 
# Data: `p3, p4, p5`

# ## Implementation

# ### MLP

# In[131]:


class MLP:
    def __init__(self, sizes, activation='sigmoid', activation_last_layer='softmax', loss='ce', learning_rate=0.01, random_seed=42):
        np.random.seed(random_seed)
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.activation = activation
        self.activation_last_layer = activation_last_layer
        self.loss = loss
        self.learning_rate = learning_rate
        self.weights = [np.random.randn(sizes[i], sizes[i-1]) / np.sqrt(sizes[i-1]) for i in range(1, self.num_layers)]
        self.biases = [np.random.randn(sizes[i], 1) for i in range(1, self.num_layers)]

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_prime(self, z):
        return np.where(z > 0, 1, 0)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def softmax_prime(self, z):
        return self.softmax(z) * (1 - self.softmax(z))

    def cross_entropy_loss(self, y, y_pred):
        m = y.shape[1]
        cost = -1/m * np.sum(y * np.log(y_pred + 1e-8))
        return np.squeeze(cost)

    def cross_entropy_loss_prime(self, y, y_pred):
        return y_pred - y

    def squared_error_loss(self, y, y_pred):
        m = y.shape[1]
        cost = 1/(2*m) * np.sum((y_pred - y)**2)
        return np.squeeze(cost)

    def squared_error_loss_prime(self, y, y_pred):
        return y_pred - y

    def forward(self, X):
        a = X.T
        for i in range(self.num_layers-1):
            z = np.dot(self.weights[i], a) + self.biases[i]
            if i == self.num_layers-2:
                if self.activation_last_layer == 'softmax':
                    a = self.softmax(z)
            else:
                if self.activation == 'sigmoid':
                    a = self.sigmoid(z)
                elif self.activation == 'relu':
                    a = self.relu(z)
                    
        return a.T

    def backward(self, X, y):
        m = X.shape[0]
        a = [X.T]
        z_s = []
        for i in range(self.num_layers-1):
            z = np.dot(self.weights[i], a[-1]) + self.biases[i]
            z_s.append(z)
            if i == self.num_layers-2:
                if self.activation_last_layer == 'softmax':
                    a.append(self.softmax(z))
            else:
                if self.activation == 'sigmoid':
                    a.append(self.sigmoid(z))
                elif self.activation == 'relu':
                    a.append(self.relu(z))
                    
        if self.loss == 'ce':
            d_a = self.cross_entropy_loss_prime(y.T, a[-1])
        elif self.loss == 'mse':
            d_a = self.squared_error_loss_prime(y.T, a[-1])
            
        d_z = d_a
        d_weights = []
        d_biases = []
        for i in range(self.num_layers-2, -1, -1):
            d_weights.insert(0, np.dot(d_z, a[i].T) / m)
            d_biases.insert(0, np.sum(d_z, axis=1, keepdims=True) / m)
            if i > 0:
                if self.activation == 'sigmoid':
                    d_z = np.dot(self.weights[i].T, d_z) * self.sigmoid_prime(z_s[i-1])
                elif self.activation == 'relu':
                    d_z = np.dot(self.weights[i].T, d_z) * self.relu_prime(z_s[i-1])
                    
        return d_weights, d_biases
    
    def train(self, X_train, Y_train, X_val, Y_val, num_epochs, batch_size): 
        n_labels = len(np.unique(Y_train))
        y_train, y_val = [np.zeros((y.shape[0], n_labels)) for y in [Y_train, Y_val]]
        for i, j in enumerate(Y_train):
            y_train[i][int(j)] = 1
        for i, j in enumerate(Y_val):
            y_val[i][int(j)] = 1   
            
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for i in range(num_epochs):
            permutation = np.random.permutation(X_train.shape[0])
            X_train = X_train[permutation, :]
            y_train = y_train[permutation]
            
            for j in range(0, X_train.shape[0], batch_size):
                
                X_batch = X_train[j : j + batch_size, :]
                y_batch = y_train[j : j + batch_size]
                
                d_weights, d_biases = self.backward(X_batch, y_batch)
                
                for k in range(len(self.weights)):
                    self.weights[k] -= self.learning_rate * d_weights[k]
                    self.biases[k] -= self.learning_rate * d_biases[k]
                    
            y_pred_train = self.forward(X_train)
            y_pred_val = self.forward(X_val)
            
            if self.loss == 'ce':
                train_loss = self.cross_entropy_loss(y_train, y_pred_train)
                val_loss = self.cross_entropy_loss(y_val, y_pred_val)
            elif self.loss == 'mse':
                train_loss = self.squared_error_loss(y_train, y_pred_train)
                val_loss = self.squared_error_loss(y_val, y_pred_val)
                
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            acc_params_train = [np.argmax(y_train, axis=1), np.argmax(y_pred_train, axis=1)]
            acc_params_val = [np.argmax(y_val, axis=1), np.argmax(y_pred_val, axis=1)]
            
            train_acc = metrics.accuracy(*acc_params_train)
            val_acc = metrics.accuracy(*acc_params_val)
            
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # train_f1 = metrics.f1Score(metrics.confusionMatrix(*acc_params_train, n_labels))
            # val_f1 = metrics.f1Score(metrics.confusionMatrix(*acc_params_val, n_labels))
            
            # print(f"Epoch {i+1}: \t  train_loss = {train_loss:.2f}  \tval_loss = {val_loss:.2f} \t  train_acc = {train_acc:.2f}  \t val_acc = {val_acc:.2f}")
        return train_losses, val_losses, train_accs, val_accs


# ### Adaboost

# In[144]:


def mlpForAdaboost(X, y, hidden_layers=[32, 16]):
    n_labels = len(np.unique(y))

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    s = int(0.7 * X.shape[0])
    xtrain, xval = X[indices[:s]], X[indices[s:]]
    ytrain, yval = y[indices[:s]], y[indices[s:]]

    layers = [xtrain.shape[1], *hidden_layers, n_labels]
    mlp = MLP(layers, 'sigmoid', 'softmax','ce', learning_rate = 0.1)
    mlp.train(xtrain, ytrain, xval, yval, num_epochs = 50 , batch_size = 256)
    return mlp

# adaboost with numpy
def adaboost(X, y, num_DT_classifiers, max_depth=10, mlp_hidden_layers=[7]):
    N = X.shape[0]
    w = np.ones(N) / N
    classifiers = []
    alphas = []
    errors = []

    # mlp as first classifier
    mlp = mlpForAdaboost(X, y, mlp_hidden_layers)
    y_pred = np.argmax(mlp.forward(X), axis=1)
    error = np.sum(w * (y_pred != y))
    alpha = np.log((1 - error) / error) / 2
    w = w * np.exp(-alpha * y * y_pred)
    w = w / np.sum(w)
    classifiers.append(mlp)
    alphas.append(alpha)
    errors.append(error)

    # decision trees are remaining classifiers
    for _ in range(num_DT_classifiers):
        tree = buildTree(X, y, gini, max_depth=max_depth)
        y_pred = predictAll(X, tree)
        error = np.sum(w * (y_pred != y))
        alpha = np.log((1 - error) / error) / 2
        w = w * np.exp(-alpha * np.array([1 if y_pred[i] == y[i] else -1 for i in range(N)]))
        w = w / np.sum(w)
        classifiers.append(tree)
        alphas.append(alpha)
        errors.append(error)

    return classifiers, alphas, errors

def predictAdaboost(X, classifiers, alphas, n_labels):
    likelihoods = np.zeros((X.shape[0], n_labels))
    y_pred = np.argmax(classifiers[0].forward(X), axis=1)
    for i in range(X.shape[0]):
        likelihoods[i][y_pred[i]] += alphas[0]
    for i in range(1, len(classifiers) - 1):
        y_pred = predictAll(X, classifiers[i])
        for j in range(X.shape[0]):
            likelihoods[j][y_pred[j]] += alphas[i]
    return np.argmax(likelihoods, axis=1)


# ## Experiment on P3 Data

# In[145]:


n_labels = len(np.unique(p3['Y']))
c, a, e = adaboost(p3['X'], p3['Y'], 10, 4)
metrics.print(predictAdaboost(p3['X'], c, a, n_labels), p3['Y'], predictAdaboost(p3['X_test'], c, a, n_labels), p3['Y_test'], visualize=True)


# In[146]:


plt.plot(np.arange(1, len(e) + 1), e)
plt.xlabel('Number of classifiers')
plt.ylabel('Error')
plt.title('Train Error vs Number of classifiers')
plt.show()


# ## Experiment on P4 Data

# In[158]:


n_labels = len(np.unique(p4['Y']))
c, a, e = adaboost(p4['X'], p4['Y'], 10, 15, [32])
metrics.print(predictAdaboost(p4['X'], c, a, n_labels), p4['Y'], predictAdaboost(p4['X_test'], c, a, n_labels), p4['Y_test'], visualize=True)


# In[159]:


plt.plot(np.arange(1, len(e) + 1), e)
plt.xlabel('Number of classifiers')
plt.ylabel('Error')
plt.title('Train Error vs Number of classifiers')
plt.show()


# ## Experiment on P5 Data

# In[150]:


n_labels = len(np.unique(p5['Y']))
c, a, e = adaboost(p5['X'], p5['Y'], 10, 4)
metrics.print(predictAdaboost(p5['X'], c, a, n_labels), p5['Y'], predictAdaboost(p5['X_test'], c, a, n_labels), p5['Y_test'], visualize=True)


# In[151]:


plt.plot(np.arange(1, len(e) + 1), e)
plt.xlabel('Number of classifiers')
plt.ylabel('Error')
plt.title('Train Error vs Number of classifiers')
plt.show()


# # Problem 4
# 
# Consider the KMNIST data and implement 
# 
#     (a) GMM-based clustering, 
#     
#     (b) K means clustering.
#     
# Evaluate and compare the Normalized Mutual Information for both algorithms. 
# 
# Experiment with different number of cluster sizes and plot the t-sne plots for all cases.
# 
# Data: `p4`

# ## Implementation

# ### GMM based clustering

# In[19]:


max_float = np.finfo("float64").max
max_exp = np.log(max_float)

def normal(x, mean ,cov):
    n = len(mean)
    val = -0.5 * (x - mean) @ np.linalg.pinv(cov) @ (x - mean)
    return np.exp(val) if np.abs(val) < max_exp else (1/max_float) / ((2 * np.pi * np.linalg.det(cov)) ** (n/2) + 1e-8)

# X -> data, k -> number of normal densities
def gmm(X, k, max_iter = 100, random_seed=42):
    # m -> number of datapoints, n -> number of features
    m, n = X.shape
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

def predictGMM(X, weights, means, covs):
    probs = np.array([[normal(x, means[i], covs[i]) * weights[i] for i in range(len(weights))] for x in X])
    return np.argmax(probs, axis=1)


# ### K-means clustering

# In[20]:


a = np.array([[1, 2], [3, 4]])
a.min(axis=0)

a[:, None] - [0, 1]


# In[21]:


# k-means clustering using numpy
def kmeans(X, k, max_iter=100, random_seed=42):
    # initialize centroids
    np.random.seed(random_seed)
    centroids = np.random.uniform(low=X.min(axis=0), high=X.max(axis=0), size=(k, X.shape[1]))
    for _ in range(max_iter):
        # calculate distance of each point from each centroid
        distances = np.linalg.norm(X[:, None] - centroids, axis=-1)
        # assign each point to the closest centroid
        clusters = np.argmin(distances, axis=-1)
        # update centroids
        centroids = np.array([np.mean(X[clusters == i], axis=0) for i in range(k)])
    return clusters, centroids

# predict cluster for each point
def predictKmeans(X, centroids):
    distances = np.linalg.norm(X[:, None] - centroids, axis=-1)
    return np.argmin(distances, axis=-1)


# ## Experiment

# In[22]:


from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.manifold import TSNE

def visualize(X, Y, k, max_iter=100, random_seed=42):
    weights, means, covs = gmm(X, k, max_iter, random_seed)
    labels_gmm = predictGMM(X, weights, means, covs)
    _, centroids = kmeans(X, k, max_iter, random_seed)
    labels_kmeans = predictKmeans(X, centroids)

    # Normalized Mutual Information
    nmi1 = normalized_mutual_info_score(Y, labels_gmm)
    nmi2 = normalized_mutual_info_score(Y, labels_kmeans)

    print(f'NMI between true labels and     GMM: {nmi1:.3f}')
    print(f'NMI between true labels and K-means: {nmi2:.3f}')

    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=random_seed)
    X_tsne = tsne.fit_transform(X)
    n_labels = len(np.unique(Y))

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(n_labels):
        ax[0].scatter(X_tsne[Y == i, 0], X_tsne[Y == i, 1], label=f'Class {i}')
    for i in range(k):
        ax[1].scatter(X_tsne[labels_gmm == i, 0], X_tsne[labels_gmm == i, 1], label=f'Cluster {i}')
        ax[2].scatter(X_tsne[labels_kmeans == i, 0], X_tsne[labels_kmeans == i, 1], label=f'Cluster {i}')
    ax[0].set_title('True labels')
    ax[1].set_title(f'GMM (NMI = {nmi1:.3f})')
    ax[2].set_title(f'K-means (NMI = {nmi2:.3f})')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.show()


# In[24]:


X, Y, X_test, Y_test, _ = trainTestSplit(p4["data"], 0.1, imgToFeatures)


# ### Number of clusters: 10

# In[184]:


visualize(X, Y, 10)


# ### Number of clusters: 5

# In[25]:


visualize(X, Y, 5)


# ### Number of clusters: 15

# In[26]:


visualize(X, Y, 15)


# ### Number of clusters: 3

# In[27]:


visualize(X, Y, 3)


# # Problem 5
# 
# Implement Principal Component Analysis on KMNIST. 
# 
# Plot the data variance as a function of the number of principal components.
# 
# Data: `p3, p4, p5`

# ## Implementation

# In[10]:


# returns eigen vectors and eigen values sorted by eigen values
def pca(X):
    m = X.shape[0]
    X = X - np.mean(X, axis=0)
    cov = np.dot(X.T, X) / (m - 1)
    eigVals, eigVecs = np.linalg.eig(cov)
    idx = eigVals.argsort()[::-1]
    eigVecs = eigVecs[:, idx]
    eigVals = eigVals[idx]
    # return np.dot(X, eigVecs[:, :k])
    return eigVecs, eigVals

def pcaTransform(X, eigVecs, k):
    return np.dot(X, eigVecs[:, :k])


# ## Data variance vs number of principal components

# In[21]:


fig, ax = plt.subplots(1, 3, figsize=(18, 5))
_, eigVals = pca(p3["X"])
eigValCumsum = np.cumsum(eigVals)
eigValCumsum = eigValCumsum / eigValCumsum[-1]
ax[0].plot(np.arange(1, len(eigVals) + 1), eigValCumsum, marker='o')
ax[0].set_xlabel("Number of components -->")
ax[0].set_ylabel("Fraction of variance retained -->")
ax[0].set_title("P3 data")

_, eigVals = pca(p4["X"])
eigValCumsum = np.cumsum(eigVals)
eigValCumsum = eigValCumsum / eigValCumsum[-1]
ax[1].plot(np.arange(1, len(eigVals) + 1), eigValCumsum, marker='o')
ax[1].set_xlabel("Number of components -->")
ax[1].set_ylabel("Fraction of variance retained -->")
ax[1].set_title("P4 data")

_, eigVals = pca(p5["X"])
eigValCumsum = np.cumsum(eigVals)
eigValCumsum = eigValCumsum / eigValCumsum[-1]
ax[2].plot(np.arange(1, len(eigVals) + 1), eigValCumsum, marker='o')
ax[2].set_xlabel("Number of components -->")
ax[2].set_ylabel("Fraction of variance retained -->")
ax[2].set_title("P5 data")

plt.show()


# In[ ]:




