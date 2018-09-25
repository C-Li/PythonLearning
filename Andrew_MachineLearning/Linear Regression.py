import numpy as np


# 假设函数
def Hypothesis(theta, X):
    y = theta.T * X
    return y


# 代价函数
def Cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)  #所有的样本
    y = np.matrix(y)
    cost = 0.5 * np.average(np.power((Hypothesis(theta, X) - y), 2))
    return cost
