import numpy as np
import matplotlib.pyplot as plt

"一元线性回归"

datafile = r'Andrew_MachineLearning\EX1_LinearRegression\ex1data1.txt'
cols = np.loadtxt(
    datafile, delimiter=',', usecols=(0, 1),
    unpack=True)  # Read in comma separated data
# loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)
# fname 要读取的文件、文件名、或生成器。
# dtype 数据类型，默认float。
# comments 注释。
# delimiter 分隔符，默认是空格。
# skiprows 跳过前几行读取，默认是0，必须是int整型。
# usecols 要读取哪些列，0是第一列。例如，usecols = （1,4,5）将提取第2，第5和第6列。默认读取所有列。
# unpack 如果为True，将分列读取。

# cols=[[x1,x2,x3,...,xm]
#       [y1,y2,y3,...,ym]]
# Form the usual "X" matrix and "y" vector
X = np.transpose(np.array(cols[:1]))  # 选择第一个元素,并转置（变成了向量）
y = np.transpose(np.array(cols[1:]))  # 选择第二个元素,并转置
m = y.size  # number of training examples

# 数据可视化
plt.figure(figsize=(10, 6))  # 建立一个宽高为10x6英寸的figure窗口实例
plt.plot(X, y, 'rx', markersize=10)  # 'rx'表示用红色的X来画点
plt.grid(True)  # 绘制网格
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.show()

# 代价函数
X = np.insert(X, 0, 1, axis=1)  # 在X向量的索引0位置插入1
# numpy.insert(arr,obj,value,axis=None)
# value 为插入的数值
# arr   为目标向量
# obj   为目标位置
# value 为想要插入的数值
# axis  为插入的维度，为None时返回一个一维数组
initial_theta = np.zeros((X.shape[1], 1))  # n行1列的向量(X有n个特征)

iterations = 1500  # 迭代（训练）次数
alpha = 0.01  # 学习率


def hypothesis(theta, X):
    h = np.dot(X, theta)
    return h


def computeCost(theta, X, y):
    cost = float((1. / (2 * m)) * np.dot((hypothesis(theta, X) - y).T,
                                         (hypothesis(theta, X) - y)))
    return cost


print(computeCost(initial_theta, X, y))  # 当theta为0时的代价=32.07


# 梯度下降
def descendGradient(X, theta_start=np.zeros(2)):
    """
    theta_start 是一个关于最初猜测的theta值的n维向量
    X           是一个m行n列的矩阵
    returns:
    theta       收敛得到的theta向量
    thetahistory theta收敛中的历史值矩阵
    jvec        代价矩阵
    """
    theta = theta_start
    jvec = []  # 代价向量
    thetahistory = []  # 保存theta的各个历史值，用于绘制路径图

    for meaninglessvariable in range(iterations):  # 迭代iterations次
        tmptheta = theta  # 建立拷贝，用于同时更新所有theta
        jvec.append(computeCost(theta, X, y))  # 计算theta对应的代价
        thetahistory.append(list(theta[:, 0]))  # 保存当前theta的第一列
        # 同时更新所有theta的值
        for j in range(len(tmptheta)):
            tmptheta[j] = theta[j] - (alpha / m) * np.sum(
                (hypothesis(initial_theta, X) - y) * np.array(X[:, j]).reshape(
                    m, 1))
        theta = tmptheta
    return theta, thetahistory, jvec


initial_theta = np.zeros((X.shape[1], 1))
theta, thetahistory, jvec = descendGradient(X, initial_theta)


# 画出代价函数的收敛图像
def plotConvergence(jvec):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(jvec)), jvec, 'bo')  # 'bo' 蓝色的实心圆画点
    plt.grid(True)
    plt.title("Convergence of Cost Function")
    plt.xlabel("Iteration number")
    plt.ylabel("Cost function")
    plt.xlim([-0.05 * iterations, 1.05 * iterations])
    # plt.ylim([4, 7]) # 此行对于这个一元线性规划的例子可以使画图更好看，但是对于多元的例子不适用
    plt.show()


plotConvergence(jvec)


# 计算预测值
def myfit(xval):
    return theta[0] + theta[1] * xval


print(X)
plt.figure(figsize=(10, 6))
plt.plot(X[:, 1], y[:, 0], 'rx', markersize=10, label='Training Data')  # 训练集
plt.plot(
    X[:, 1],        # x值
    myfit(X[:, 1]),  # 预测的y值
    'b-',           # 蓝色实线
    label='Hypothesis: h(x) = %0.2f + %0.2fx' % (theta[0], theta[1]))   # 预测函数图像
plt.grid(True)
plt.ylabel('Profit in $10,000s (y)')
plt.xlabel('Population of City in 10,000s (x)')
plt.legend()
plt.show()

"梯度下降的可视化"

# 绘制3d图需要的包
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import itertools

fig = plt.figure(figsize=(10, 7))
ax = fig.gca(projection='3d')

xvals = np.arange(-10, 10, .5)
yvals = np.arange(-1, 4, .1)
myxs, myys, myzs = [], [], []
for david in xvals:
    for kaleko in yvals:
        myxs.append(david)  # x
        myys.append(kaleko)  # y
        myzs.append(computeCost(np.array([[david], [kaleko]]), X, y))   # 代价

scat = ax.scatter(
    myxs, myys, myzs,   # x,y,z
    c=np.abs(myzs),     # 颜色
    cmap=plt.get_cmap('YlOrRd'))  # 'YlOrRd' 黄色到红色的渐变

plt.xlabel(r'$\theta_0$', fontsize=20)  # $包围的是TeX 方程表达式，\theta_0表示带下标的theta
plt.ylabel(r'$\theta_1$', fontsize=20)
plt.title('Cost (Minimization Path Shown in Blue)', fontsize=20)
plt.plot([x[0] for x in thetahistory], [x[1] for x in thetahistory], jvec,
         'bo-')  # 'bo-'蓝色的实心圆并连线
plt.show()


"多元线性规划"

# 数据导入
datafile = r'Andrew_MachineLearning\EX1_LinearRegression\ex1data2.txt'
cols = np.loadtxt(
    datafile, delimiter=',', usecols=(0, 1, 2),
    unpack=True)
X = np.transpose(np.array(cols[:-1]))
y = np.transpose(np.array(cols[-1:]))
m = y.size
X = np.insert(X, 0, 1, axis=1)

# 数据可视化
plt.grid(True)
plt.xlim([-100, 5000])
dummy = plt.hist(X[:, 0], label='col1')
dummy = plt.hist(X[:, 1], label='col2')
dummy = plt.hist(X[:, 2], label='col3')
plt.title('Clearly we need feature normalization.')
plt.xlabel('Column Value')
plt.ylabel('Counts')
dummy = plt.legend()
plt.show()

# Feature normalizing the columns (subtract mean, divide by standard deviation)
# 特征正规化（归一化）（均值减法，再除以标准差）
# 保存平均值和标准差
# 注意，不要修改原X矩阵，应当使用它的拷贝
stored_feature_means, stored_feature_stds = [], []
Xnorm = X.copy()
for icol in range(Xnorm.shape[1]):  # Xnorm.shape[1]返回Xnorm的列数，这里是3
    stored_feature_means.append(np.mean(Xnorm[:, icol]))    # 平均值
    stored_feature_stds.append(np.std(Xnorm[:, icol]))      # 标准差
    # 跳过第一列(全为1)
    if not icol:
        continue
    # 使用保存的平均值和标准差而不是重新计算，这样更快
    Xnorm[:, icol] = (Xnorm[:, icol] - stored_feature_means[-1]
                      ) / stored_feature_stds[-1]

# 将正规化后的数据可视化
plt.grid(True)
plt.xlim([-5, 5])
dummy = plt.hist(Xnorm[:, 0], label='col1')
dummy = plt.hist(Xnorm[:, 1], label='col2')
dummy = plt.hist(Xnorm[:, 2], label='col3')
plt.title('Feature Normalization Accomplished')
plt.xlabel('Column Value')
plt.ylabel('Counts')
dummy = plt.legend()
plt.show()

# 运行多元梯度下降，初始theta仍为0
# (注意! 在正归化之前不可用，可能会溢出)
initial_theta = np.zeros((Xnorm.shape[1], 1))
theta, thetahistory, jvec = descendGradient(Xnorm, initial_theta)

# 画出代价函数的收敛图像
plotConvergence(jvec)

# 输出theta
print("theta is :\n", theta)

ytest = np.array([1650., 3.])
# To "undo" feature normalization, we "undo" 1650 and 3, then plug it into our hypothesis
# 为了计算预测值，使用正规化之前的数据[1650,3]代入预测函数
ytestscaled = [
    (ytest[x] - stored_feature_means[x + 1]) / stored_feature_stds[x + 1]
    for x in range(len(ytest))
]
ytestscaled.insert(0, 1)
# 输出梯度下降算法获得的模型对于[1650,30]的预测结果
print("Check of result: What is price of house with 1650 square feet and 3 bedrooms?")
print("$%0.2f" % float(hypothesis(theta, ytestscaled)))

from numpy.linalg import inv    # 用于计算逆矩阵


# 实现正规方程
def normEqtn(X, y):
    #restheta = np.zeros((X.shape[1],1))
    return np.dot(np.dot(inv(np.dot(X.T, X)), X.T), y)


# 输出正规方程对于[1650,30]的预测结果
print("Normal equation prediction for price of house with 1650 square feet and 3 bedrooms")
print("$%0.2f" % float(hypothesis(normEqtn(X, y), [1, 1650., 3])))
