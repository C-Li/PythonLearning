import matplotlib.pyplot as plt
import numpy as np
'''
格式字符串 默认是b- 蓝色实线
形状
    '-'	solid line style
    '--'dashed line style
    '-.'dash-dot line style
    ':'	dotted line style
    '.'	point marker
    ','	pixel marker
    'o'	circle marker
    'v'	triangle_down marker
    '^'	triangle_up marker 
    '<'	triangle_left marker
    '>'	triangle_right marker
    '1'	tri_down marker
    '2'	tri_up marker
    '3'	tri_left marker
    '4'	tri_right marker
    's'	square marker
    'p'	pentagon marker
    '*'	star marker
    'h'	hexagon1 marker
    'H'	hexagon2 marker
    '+'	plus marker
    'x'	x marker
    'D'	diamond marker
    'd'	thin_diamond marker
    '|'	vline marker
    '_'	hline marker
颜色
    ‘b’	blue
    ‘g’	green
    ‘r’	red
    ‘c’	cyan
    ‘m’	magenta
    ‘y’	yellow
    ‘k’	black
    ‘w’	white
'''

'使用数组作为参数'
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')  # x集合,y集合,格式字符串(红色实心点)
plt.axis([0, 6, 0, 20])  # [xmin，xmax，ymin，ymax] 指定轴域的可视区域
plt.show()

'plot也能使用numpy数组作为参数'

# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

# 三条线 红色虚线 蓝色正方形 和绿色三角形
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()

'控制线条属性'
# 线条有许多你可以设置的属性：linewidth，dash style，antialiased等。有几种方法可以设置线属性：
x = x1 = t
y = t**2
y1 = t**3
# 使用关键字参数：
plt.plot(x, y, linewidth=2.0)

# 使用Line2D实例的setter方法。 plot返回Line2D对象的列表，例如line1,line2 = plot(x1，y1，x2，y2)。
# 在下面的代码中，我们假设只有一行，返回的列表长度为 1。我们对line使用元组解构，得到该列表的第一个元素：
line, = plt.plot(x, y, '-')
line.set_antialiased(False)  # turn off antialising

# 使用setp()命令。 下面的示例使用 MATLAB 风格的命令来设置线条列表上的多个属性。
# setp使用对象列表或单个对象透明地工作。 你可以使用 python 关键字参数或 MATLAB 风格的字符串/值对：
lines = plt.plot(x, y, x1, y1)
plt.setp(lines, color='r', linewidth=2.0)  # 使用关键字参数
plt.setp(lines, 'color', 'r', 'linewidth', 2.0)  # 或者 MATLAB 风格的字符串值对
'''
下面是可用的Line2D属性。
    属性	            值类型
    alpha	            浮点值
    animated	        [True / False]
    antialiased or aa	[True / False]
    clip_box	        matplotlib.transform.Bbox 实例
    clip_on	            [True / False]
    clip_path	        Path 实例， Transform，以及Patch实例
    color or c	        任何 matplotlib 颜色
    contains	        命中测试函数
    dash_capstyle	    ['butt' / 'round' / 'projecting']
    dash_joinstyle	    ['miter' / 'round' / 'bevel']
    dashes	            以点为单位的连接/断开墨水序列
    data	            (np.array xdata, np.array ydata)
    figure	            matplotlib.figure.Figure 实例
    label	            任何字符串
    linestyle or ls	    [ '-' / '--' / '-.' / ':' / 'steps' / ...]
    linewidth or lw	    以点为单位的浮点值
    lod	                [True / False]
    marker	            [ '+' / ',' / '.' / '1' / '2' / '3' / '4' ]
    markeredgecolor or mec	任何 matplotlib 颜色
    markeredgewidth or mew	以点为单位的浮点值
    markerfacecolor or mfc	任何 matplotlib 颜色
    markersize or ms	浮点值
    markevery	        [ None / 整数值 / (startind, stride) ]
    picker	            用于交互式线条选择
    pickradius	        线条的拾取选择半径
    solid_capstyle	    ['butt' / 'round' / 'projecting']
    solid_joinstyle	    ['miter' / 'round' / 'bevel']
    transform	        matplotlib.transforms.Transform 实例
    visible	            [True / False]
    xdata	            np.array
    ydata	            np.array
    zorder	            任何数值
'''

'处理多个图形和轴域'

# MATLAB 和 pyplot 具有当前图形和当前轴域的概念。 所有绘图命令适用于当前轴域。
# 函数gca()返回当前轴域（一个matplotlib.axes.Axes实例），gcf()返回当前图形（matplotlib.figure.Figure实例）。
# 通常，你不必担心这一点，因为它都是在幕后处理。 下面是一个创建两个子图的脚本。

def f(t):
    return np.exp(-t) * np.cos(2 * np.pi * t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(1)   # 这句是可选的，因为默认情况下将创建figure(1)

# subplot()命令指定numrows，numcols，fignum，其中fignum的范围是从1到numrows * numcols。 
# 如果numrows * numcols <10，则subplot命令中的逗号是可选的。

plt.subplot(211)    # 如果不手动指定任何轴域，则默认创建subplot(111)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212)    # 也可以写成plt.subplot(2,1,2) 
plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')

plt.show()

# 在一个图形用close()显式关闭之前，该图所需的内存不会完全释放。 
# 在调用close()之前，pyplot会维护内部引用。

'添加文本'

# text()命令可用于在任意位置添加文本，xlabel()，ylabel()和title()用于在指定的位置添加文本
# text（）      在Axes的任意位置添加文本;  matplotlib.axes.Axes.text（）。
# xlabel（）    在x轴上添加标签;          matplotlib.axes.Axes.set_xlabel（）。
# ylabel（）    在y轴上添加标签;          matplotlib.axes.Axes.set_ylabel（）。
# title（）     为Axes添加标题;           matplotlib.axes.Axes.set_title（）。
# figtext（）   在图中的任意位置添加文本;  matplotlib.figure.Figure.text（）。
# suptitle（）  为图添加标题;             matplotlib.figure.Figure.suptitle（）。
# annotate（）  添加注释 可选箭头，到轴;   matplotlib.axes.Axes.annotate（）。

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# 数据的直方图
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()

'在文本中使用表达式'
# matplotlib有一个内置的 TeX 表达式解析器和布局引擎，并且自带了自己的数学字体
# matplotlib在任何文本表达式中接受 TeX 方程表达式。 例如，要在标题中写入表达式，可以编写一个由美元符号包围的 TeX 表达式：
plt.title(r'$\sigma_i=15$')