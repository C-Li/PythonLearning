import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

'将图像数据导入到 NumPy 数组'

img = mpimg.imread(r'MatplotlibDemo\stinkbug.png')
print('图片数组'+str(img.shape)+'内容为：') # 这里是个灰度图，img是一个二维数组
print(img)

# Matplotlib 已将每个通道的8位数据重新定标为 0.0 和 1.0 之间的浮点数。 
# 作为旁注，Pillow 可以使用的唯一数据类型是uint8。 
# Matplotlib 绘图可以处理float32和uint8，但是对于除 PNG 之外的任何格式的图像，读取/写入仅限于uint8数据。

# 每个内部列表表示一个像素。 这里，对于 RGB 图像，有 3 个值。 由于它是一个黑白图像，R，G 和 B 都是类似的。 
# RGBA（其中 A 是阿尔法或透明度）对于每个内部列表具有 4 个值，而且简单亮度图像仅具有一个值（因此仅是二维数组，而不是三维数组）。 
# 对于 RGB 和 RGBA 图像，matplotlib支持float32和uint8数据类型。 对于灰度，matplotlib只支持float32。

'将 NumPy 数组绘制为图像'

imgplot = plt.imshow(img)   # 也可以绘制任何 NumPy 数组。
plt.show()

'对图像绘图应用伪彩色方案'
# 伪彩色可以是一个有用的工具，用于增强对比度和更易于可视化你的数据。伪彩色仅与单通道，灰度，亮度图像相关。
# 默认颜色表（也称为查找表，LUT）。 默认值称为jet。 有很多其他方案可以选择。
plt.imshow(img, cmap="hot")
plt.show()
# 还可以使用set_cmap()方法更改现有绘图对象上的颜色
imgplot = plt.imshow(img)
imgplot.set_cmap('nipy_spectral')
plt.show()


'颜色刻度参考'
# 了解颜色代表什么值对我们很有帮助。 我们可以通过添加颜色条来做到这一点。
imgplot = plt.imshow(img,cmap='nipy_spectral')
plt.colorbar()
plt.show()

'检查特定数据范围'
# 有时，你想要增强图像的对比度，或者扩大特定区域的对比度，同时牺牲变化不大，或者无所谓的颜色细节。 
# 找到有趣区域的最好工具是直方图。 要创建我们的图像数据的直方图，我们使用hist()函数。
plt.hist(img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
plt.show()
# 通常，图像的『有趣』部分在峰值附近，你可以通过剪切峰值上方和/或下方的区域获得额外的对比度。
# 我们通过将clim参数传递给imshow来实现。 你也可以通过对图像绘图对象调用set_clim()方法来做到这一点。
plt.figure(1)

plt.subplot(121)
plt.imshow(img)
plt.title("处理前")

plt.subplot(122)
plt.title("处理后")
imgplot = plt.imshow(img, clim=(0.0, 0.7))
plt.show()

'数组插值方案'
# 插值根据不同的数学方案计算像素『应有』的颜色或值。 发生这种情况的一个常见的场景是调整图像的大小。 像素的数量会发生变化，但你想要相同的信息。 由于像素是离散的，因此存在缺失的空间。 插值就是填补这个空间的方式。 这就是当你放大图像时，你的图像有时会出来看起来像素化的原因。 当原始图像和扩展图像之间的差异较大时，效果更加明显。 让我们加载我们的图像并缩小它。 我们实际上正在丢弃像素，只保留少数几个像素。 现在，当我们绘制它时，数据被放大为你屏幕的大小。 由于旧的像素不再存在，计算机必须绘制像素来填充那个空间。
# 我们将使用用来加载图像的 Pillow 库来调整图像大小。
from PIL import Image

img = Image.open(r'MatplotlibDemo\stinkbug.png') # 这里跟前面不同了
img.thumbnail((64, 64), Image.ANTIALIAS) # 先把图片变小 antialias 平滑滤波插值，用于从大图获取小图
print(img.format, img.size, img.mode)
# 可以转换成九种不同模式，分别为1，L，P，RGB，RGBA，CMYK，YCbCr，I，F
# img.convert("1") 二值图像，非黑即白
# L         8位灰度图
# P         8位彩色图
# RGB       24位彩色图像
# RGBA      32位彩色图像，比RGB多一个alpha通道
# CMYK      32位彩色图像，印刷四分色模式
# YCbYCr    24位彩色图像，Y是指亮度分量，Cb指蓝色色度分量，而Cr指红色色度分量。人的肉眼对视频的Y分量更敏感，因此在通过对色度分量进行子采样来减少色度分量后，肉眼将察觉不到的图像质量的变化。
# I         32位整型灰色图像
# F         32位浮点灰色图像

img=np.array(img)   # 不转换成numpy数组会出错

# 默认插值，双线性
plt.title("bilinear")
imgplot = plt.imshow(img)
plt.show()
# 最邻近插值
plt.title("nearest")
imgplot = plt.imshow(img, interpolation="nearest")
plt.show()
# 双立方插值
imgplot = plt.imshow(img, interpolation="bicubic")
plt.title("bicubic")
plt.show()