import numpy as np

'Numpy基础'
x=np.array([1,2,3]) #定义一个一维三元的数组
print(x[0])         #输出第一个元素 1
x[0]=0              #修改第一个元素为 0

# 高维数组
#[[1,2,3]
# [4,5,6]
# [7,8,9]]
x=np.array([[1,2,3],[4,5,6],[7,8,9]])
# 应对高维数组的索引方法
print(x[:,0])   # 输出[1 4 7]
print(x[:,1])   # 输出[2 5 8]
print(x[2,:])   # 输出[7 8 9]
print(x[:2,0])  # 输出[1 4]
print(x[1,:2])  # 输出[4 5]
print(x[:2,:2]) # 输出[[1 2],[4 5]]

# 初始化数组
x=np.zeros((3,3))           # 全0的3x3数组
x=np.ones((3,3))            # 全1的3x3数组
x=np.random.random((3,3))   # 随机的3x3数组

# 数组运算
x=np.array([[1,2],[3,4]])
y=np.array([[4,3],[2,1]])
# 加
print(x+y)  # [[5,5],[5,5]]
# 减
print(x-y)  # [[-3，-1]，[1,3]]
# 乘
print(x*y)          # element-wise乘法 [[4,6],[6,4]]
print(np.dot(x,y))  # 矩阵乘法 [[8,5],[20,13]]
# 除
print(x/y)  # [[0.25,2/3],[1.5,4]]

# 对高维数组的数学运算操作
#[[1,2]
# [3,4]]
# 对每个元素取e的指数
print(np.exp(x))            # [[e^1,e^2],[e^3,e^4]]
# 对每个元素取根号
print(np.sqrt(x))           # [[1,sqrt(2)],[sqrt(3),2]]
# 对第一个axis取均值（1和3,2和4）
print(np.average(x,axis=0)) # [2,3]
# 对第二个axis取均值（1和2,3和4）
print(np.average(x,axis=1)) # [1.5,3.5]
# axis：矩阵的维度。可以视为不同深度的for循环所对应的数据

'向量化与升维'
# 将for循环替换成Numpy运算
# 将难以直接向量化的算法所对应的数组进行升维

x=np.random.random((1000,1000))
ans=np.zeros((1000,1000))
# 目的：计算x+1并将结果存进ans中
# for循环写法(545ms)：
for i,row in enumerate(x):
    for j,elem in enumerate(row):
        ans[i][j]=elem+1
# Numpy运算写法(4.6ms)：
ans=x+1
# 更快，更省内存的写法(2.57ms)
np.add(x,1,ans)

'Numpy的一个应用思想'
# 尽量避免不必要的拷贝，包括但不限于：
# x=x+1(建议使用x+=1)
# y=x.flatten()(建议使用y=x.ravel())
# x=x.T(没有替代方案，尽量少用转置)