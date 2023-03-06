import cv2
import scipy.io as scio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
 

 # 数据矩阵转图片的函数
def MatrixToImage(data):
    data = data * 255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im
 

 
 # 添加路径，文件夹下存放多个.mat文件
# 两个文件夹一个放.mat,一个放转换好的图片
datafolder = r'/Users/wanghaihang/Downloads/MtS-WH/test/'
savefolder = r'/Users/wanghaihang/Downloads/MtS-WH/test/'
path = os.listdir(datafolder)
print(os.path.splitext('100007.mat'))
# 这一步用于测试你是否正确设置了路径，也展示了splitxt函数的结果

 # 先取一个文件做实验，接下来在控制台一行一行运行
each_mat='hanyang20009label.mat'
first_name, second_name = os.path.splitext(each_mat)
    # 拆分.mat文件的前后缀名字，
 
each_mat = os.path.join(datafolder, each_mat)
    # print(each_mat)
    # 校验步骤，输出应该是路径
 
array_struct = scio.loadmat(each_mat)
print(array_struct)
    # 校验步骤，输出的应该是一个结构体，然后查看你的控制台，看数据被存在了哪个字段里
    # 我的数据被放在了result里，所以下面填result
 
array_data = array_struct['hanyang2009']  # 取出需要的数字矩阵部分
    # print(array_data)
    # 校验步骤，看是否正常读出了数据
 
new_im = MatrixToImage(array_data)  # 调用函数
plt.imshow(array_data, cmap=plt.cm.gray, interpolation='nearest')
new_im.show()
new_im.save(savefolder+first_name + '.jpg')  # 保存图片

