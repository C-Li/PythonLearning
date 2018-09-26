# 首先在扩展库中搜索安装Python和Jupyter插件
# 输入#%%插入新的单元格（相当于Jupyter中的In[0]之类的格子）
# 点击#%%上方的Run cell 运行该段代码，右侧会出现一个新窗口显示输出

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

#%%
x = np.linspace(0, 20, 100)
plt.plot(x, np.sin(x))

#%%
print("苟利国家生死以，岂因祸福避趋之")

#%%
plt.title("Cos(x) plot")
plt.plot(x,np.cos(x),"r-")

