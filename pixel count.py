from PIL import Image


image = Image.open('D:\wp\Dataset2023.1.2\output2/segmentation-21349.png')  # 图片的路径

a, b = image.size
num = 0
for i in range(a):
    for j in range(b):
        pixel = image.getpixel((i, j))  # 读取该点的像素值
        if pixel == 255:
            num += 1
image.show()
print('pixel number of areas:', num)