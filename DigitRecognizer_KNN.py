#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, decomposition


# Loading data
X = np.loadtxt("train.csv", delimiter=',', skiprows=1, usecols=range(1,785,1))
y = np.loadtxt("train.csv", delimiter=',', skiprows=1, usecols=0)

print("Input size: ")
print(X.shape)


# Dividing data into train and validation
x_val = X[33600:42000, :]  # Validation set of size 8400 x 784, from row 33600:42000 (exclusive)
y_val = y[33600:42000, ]   # Validation lables of size 8400 x 1, from row 33600:42000 (exclusive)
print(x_val.shape)  # Confirm the shape

x_tr = X[0:33600, :]  # training examples of size 33600 x 784
y_tr = y[0:33600]     # training labels
print(x_tr.shape)     # confirm the shape


print("============= Experiment #1 =============")
print("#1: Testing with uniform weights (All points in each neighborhood are weighted equally): ")
for k in range (1,16,2):
    clf = neighbors.KNeighborsClassifier(k, weights="uniform")
    clf.fit(x_tr, y_tr)
    print("K=", k, "   |   Accuracy= ", clf.score(x_val,y_val))

print("#2: Testing with distance weights (weight points based on their distance): ")
for k in range (1,16,2):
    clf = neighbors.KNeighborsClassifier(k, weights="distance")
    clf.fit(x_tr, y_tr)
    print("K=", k, "   |   Accuracy= ", clf.score(x_val,y_val))


# Applying PCA:

print("============= Experiment #2: apply PCA =============")
significance_array = []
for i in range(25,785, 25):
    pca_unit = decomposition.PCA(n_components = i)
    pca_unit.fit(x_tr)
    info = sum(pca_unit.explained_variance_ratio_)
    significance_array.append(info)
    print("number of features: ", i, " | amount of captured information: ",info)

plt.plot(range(25,785, 25), significance_array)
plt.xlabel("Number of features to be kept")
plt.ylabel("Total significance")


# Testing the accuracy obtained by different number of features

print("============= Experiment #3 =============")
accuracy_array = []
index = 0
for i in range(50,200, 25):
    pca_unit = decomposition.PCA(i)
    pca_unit.fit(x_tr)
    transformed_x_tr= pca_unit.transform(x_tr)
    transformed_x_val= pca_unit.transform(x_val)
    print("======== Testing with", i, " features ========")
    accuracy_array.append(list())
    for k in range (1,12,2):
        clf = neighbors.KNeighborsClassifier(k, weights="distance")
        clf.fit(transformed_x_tr, y_tr)
        accuracy = clf.score(transformed_x_val,y_val)
        accuracy_array[index].append(accuracy)
        print("k = ", k, "Accuracy = ", accuracy)
    index=index+1

acc = []
for accuracies in accuracy_array:
    acc.append(max(accuracies))
plt.plot(list(range(50,200, 25)), acc)
plt.xlabel("Number of features")
plt.ylabel("Maximum accuracy")





