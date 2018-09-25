import numpy as np
'''
numpy中数组和矩阵的区别：
matrix是array的分支，matrix和array在很多时候都是通用的，你用哪一个都一样。
官方建议大家如果两个可以通用，那就选择array，因为array更灵活，速度更快，很多人把二维的array也翻译成矩阵。
matrix的优势是相对简单的运算符号，比如两个矩阵相乘，就是用符号*，但是array相乘不能这么用，得用方法.dot()
array的优势是不仅仅表示二维，还能表示3、4、5…维，而且在大部分Python程序里，array也是更常用的。
'''

# matrix 矩阵

A = np.matrix('1.0 2.0; 3.0 4.0')
B = np.matrix([[1.0, 2.0], [3.0, 4.0]])  # 和上面等效
print(A)
#[[ 1.  2.]
# [ 3.  4.]]
print(B)
#[[ 1.  2.]
# [ 3.  4.]]

print(A.T)  # transpose 转置    修改A.T会影响到A
#[[ 1.  3.]
# [ 2.  4.]]

X = np.matrix('5.0 7.0')
Y = X.T
print(Y)
#[[5.]
# [7.]]

print(A * Y)  # matrix multiplication 矩阵乘法 矢量积 叉乘
#[[19.]
# [43.]]

print(np.multiply(A, Y))  # 数量积 点乘
#[[ 5. 10.]
# [21. 28.]]

print(A**2)  # = A*A 矢量积
# 当A是Array时，A**2 则是数量积 元素分别平方

print(np.power(A, 2))  # 数量积 元素分别平方
print(np.square(A))  # 数量积 元素分别平方

print(A.I)  # inverse 求逆矩阵
#[[-2.   1. ]
# [ 1.5 -0.5]]

print(np.linalg.solve(A, Y))  # solving linear equation 求解线性矩阵方程
#[[-3.],
# [ 4.]]

# 切片
M = np.matrix([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
M.shape = (3, 4)  # 3x4的矩阵，3行4列

print(M[:])
print(M[:].shape)
#[[ 0  1  2  3]
# [ 4  5  6  7]
# [ 8  9 10 11]]
#(3, 4)

print(M[:, 1])
print(M[:, 1].shape)  #截取第1列（从0开始）
#[[1]
# [5]
# [9]]
#(3, 1)

print(M[:, [1, 3]])
print(M[:, [1, 3]].shape)  # 截取第1列和第3列
#[[ 1  3]
# [ 5  7]
# [ 9 11]]
#(3, 2)

# vector 向量
V = np.matrix('1;2;3')
print(V)
#[[1]
# [2]
# [3]]
print(V[1])
#[[2]]
