# Anaconda 几乎把所有常用且优异的科学计算库都集成在了一起

# Anaconda操作：

# 查看帮助
'conda -h'

# 包
# 安装一个包
'conda install package_name'    # 也可以同时安装几个包，用空格隔开
# 指定安装包的版本
'conda instal numpy=1.10'
# 移除一个包
'conda remove package_name'
# 升级包版本
'conda update package_name'
# 查看所有包
'conda list'
# 模糊查询
'conda search search_term'

# 环境
# 基于Python3.6版本建立一个名为python36的环境
'conda create --name python36 python=3.6'
# 激活此环境
'active python36'   # Windows
'source active python36'    # Linux
# 检查Python版本
'python -V'
# 退出当前环境
'decactive python36'
# 删除该环境
'conda remove -n python36 --all'
'conda env remove -n python36'  # 另一种方法


