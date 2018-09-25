import numpy as np
'''
NumPy的数组类被称作ndarray。通常被称作数组。
注意numpy.array和标准Python库类array.array并不相同，后者只处理一维数组和提供少量功能。
'''

data = np.array([1, 2, 3])
print(data)
data.shape  # 数组的维度，如（2,3）
data.dtype  # 数组中元素的数据类型
data.ndim  # 数组轴的个数(秩)
data.size  # 数组元素的总个数
data.itemsize  # 数组中每个元素的字节大小
data.data  # 包含实际数组元素的缓冲区，通常用不到

# 创建数组

data = []
arr = np.array(data)  # 传递一个list对象创建ndarray对象

a = np.array([2, 3, 4])
b = np.array([(1.5, 2, 3), (4, 5, 6)])  # 二维数组
c = np.array([[1, 2], [3, 4]], dtype=complex)  #指定数据类型为虚数
print(c)

d = np.zeros(10)  # 创建长度为10的全0数组
print(d)
d = np.zeros((3, 6))  # 创建3行6列的全0二维数组
print(d)
d = np.empty((2, 3))  # 创建2行3列的内容随机二维数组
print(d)
d = np.arange(5, 25, 5)  # 对应python内置的range函数（开始，结束，步长）
print(d)
# [5 10 5 20]
d = np.arange(5)  # 对应python内置的range函数（从0开始，步长为1）
print(d)
# [0 1 2 3 4]

# 基本运算
# 大小相等的数组之间，任何算数运算都会将运算应用到元素级。

a = np.array([20, 30, 40, 50])
b = np.arange(4)

print(a - b)
# [20  29  38  47]

print(b**2)
# [0  1  4  9]

print(10 * np.sin(a))
#[ 9.12945251 -9.88031624 7.4511316 -2.62374854]

print(a < 35)
#[True True False False]

A = np.array([[1, 1], [0, 1]])
B = np.array([[2, 0], [3, 4]])

print(A * B)  # elementwise product 数量积
#[[2 0]
# [0 4]]

print(np.dot(A, B))  # matrix product   矩阵乘法 向量积
#[[5 4]
# [3 4]]

C = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#[[1,2,3]
# [4,5,6]
# [7,8,9]]

# 数组索引
# 负数表示从后往前

# 一维数组 略
# 多维数组
print(C[0, 0])  # 输出A的第一行第一列的值
#         行号    列号
print(C[[0, 2], [0, 2]])  # 选择A的第一行和第三行，然后选择第一行的第一列和第三行的第三列输出
# 布尔索引
print(C[C > 5])  # 输出大于5的元素(一维数组)
# 布尔索引中还可以使用：
# numpy.iscomplex() 复数
# numpy.isreal() 实数
# numpy.isfinite() 有限
# numpy.isinf() 无穷大
# numpy.isnan() NotANumber


# 数组切片
# 切片得到的是原多维数组的一个视图(view) ，修改切片中的内容会导致原多维数组的内容也发生变化
# 由 start, stop, step 三个部分组成
# 负数表示从后往前

# 一维数组
arr = np.arange(12)

print(arr[:4])  #从索引0到索引4
print(arr[7:10])    #从索引7到索引10
print(arr[0:12:4])  #从索引0到索引12，步长为4

# 多维数组 用","隔开维度就可以了

arr = np.arange(12).reshape((3, 4))
#[[0,1,2,3]
# [4,5,6,7]
# [8,9,10,11]]

# 取第一维的索引 1 到索引 2 之间的元素，也就是第二行
# 再从中取第二维的索引 1 到索引 3 之间的元素，也就是第二列和第三列
print(arr[1:2, 1:3])
# [[5,6]]

# 取第一维的全部（选中全部三行）
# 从中按步长为 2 取第二维的索引 0 到末尾 之间的元素，也就是第一列和第三列
print(arr[:, ::2])
#[[0,2]
# [4,6]
# [8,10]]

# 取第一列，输出为一维数组（索引）
print(arr[:,0])
# [0,4,8]

# 取第一列（切片）
print(arr[:,:1])
# [[0]
#  [4]
#  [8]]

# 对于维数超过 3 的多维数组，还可以通过 '…' 来简化操作
arr = np.arange(24).reshape((2, 3, 4))
print(arr[1, ...])  # 等价于 arr[1, :, :]
print(arr[..., 1])  # 等价于 arr[:, :, 1]
