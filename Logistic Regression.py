import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 代价函数
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-1 * y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))


# 正则化后的代价函数，learningRate是正则化参数lambda
def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-1 * y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X)) * np.sum(
        np.power(theta[:, 1:theta.shape[1]], 2)))
    return np.sum(first - second) / (len(X)) + reg