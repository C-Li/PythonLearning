import matplotlib.pyplot as plt

show_height = 50
show_width = 80

ascii_char = list(
    r"$@B%8&WM#*oahkbdpqwmZO)QLCJUYXzcvunxrjft/\|()1{}[]?-_+`<>i!lI;:,\"^`.")

char_len = len(ascii_char)

# 使用imread来读取图像，对于彩图，返回size=height*width*3的图像
# matplotlib中色彩排列为RGB
# opencv的cv2中色彩排列是BGR
pic = plt.imread("wm.jpg")

# 获取图像高宽
pic_height, pic_width, _ = pic.shape

# RGB转灰度的公式
gray = 0.2126 * pic[:, :, 0] + 0.7152 * pic[:, :, 1] + 0.0722 * pic[:, :, 2]

# 直接显示灰度图
plt.imshow(gray,cmap="gray")
plt.show()

# 根据灰度值映射到相应的ascii_char
for i in range(show_height):
    # 根据比例映射到对应的像素
    y = int(i * pic_height / show_height)
    text = ""
    for j in range(show_width):
        x = int(j * pic_width / show_width)
        text += ascii_char[int(gray[y][x] / 256 * char_len)]
    print(text)
