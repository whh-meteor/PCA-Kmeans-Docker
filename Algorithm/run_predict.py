#https://appliedmachinelearning.wordpress.com/2017/11/25/unsupervised-changed-detection-in-multi-temporal-satellite-images-using-pca-k-means-python-code/
from cgitb import reset
from crypt import methods
from email.mime import image
from flask import Flask
from flask import request
from flask import  Response,send_file,jsonify
from PIL import Image
import base64
import io
from flask import render_template
import psutil
 
 
# 从别的文件里引用，实例化一个类，方便下面调用
# from src import yingzhibiao_predict
# p = yingzhibiao_predict.predictt()
from uuid import uuid4
import matplotlib.pyplot as plt 
import os
app = Flask(__name__)

 
import cv2
import numpy as np

from numpy import (amin, amax, ravel, asarray, arange, ones, newaxis,
                   transpose, iscomplexobj, uint8, issubdtype, array)


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
# from scipy.misc import imread , imresize, imsave 

import imageio
# from imageio import imread

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('test.html')

@app.route('/PCA_KMeans', methods=['GET', 'POST'])
def PCA_KMeans():
    print("2--data内存使用" , get_current_memory_gb())
    data = request.form
    print(data)
    imagepath1 = request.files['img1']
    imagepath2 = request.files['img2']
    tmp_fname1 = os.path.join('Caching', uuid4().__str__()+'.bmp')#？？？？？
    tmp_fname2 = os.path.join('Caching', uuid4().__str__()+'.bmp')
    imagepath1.save(tmp_fname1)
    imagepath2.save(tmp_fname2)

    print('Operating')
    
    image1 = imread(imagepath1)[:, :, 0:3]
    image2 = imread(imagepath2)[:, :, 0:3]
    print("3--read内存使用" , get_current_memory_gb(),'MB')
    print(image1.shape,image2.shape) 
    new_size = np.asarray(image1.shape) /5
    new_size = new_size.astype(int) *5
    image1 = imresize(image1, (new_size)).astype(np.int16) 
    image2 = imresize(image2, (new_size)).astype(np.int16)
    print("3.5--内存使用" , get_current_memory_gb(),'MB')
    diff_image = abs(image1 - image2)   
    #imsave('diff.jpg', diff_image)
    print("4--diff内存使用" , get_current_memory_gb(),'MB')
    imageio.imwrite('./Results/diff.jpg', diff_image)
    print('\nBoth images resized to ',new_size)
    
    vector_set, mean_vec = find_vector_set(diff_image, new_size)
    
    pca     = PCA()
    pca.fit(vector_set)
    EVS = pca.components_
        
    FVS     = find_FVS(EVS, diff_image, mean_vec, new_size)
    
    print('\ncomputing k means')
    
    components = 3
    least_index, change_map = clustering(FVS, components, new_size)
    
    change_map[change_map == least_index] = 255
    change_map[change_map != 255] = 0
    
    change_map = change_map.astype(np.uint8)
    kernel     = np.asarray(((0,0,1,0,0),
                             (0,1,1,1,0),
                             (1,1,1,1,1),
                             (0,1,1,1,0),
                             (0,0,1,0,0)), dtype=np.uint8)
    cleanChangeMap = cv2.erode(change_map,kernel)
    imageio.imwrite("./Results/t1.jpg", image1)
    imageio.imwrite("./Results/t2.jpg", image2)
    imageio.imwrite("./Results/changemap.jpg", change_map)
    imageio.imwrite("./Results/cleanchangemap.jpg", cleanChangeMap)   

    T1_Steam = return_img_stream("./Results/t1.jpg")
    T2_Steam = return_img_stream("./Results/t2.jpg")
    img_stream = return_img_stream("./Results/diff.jpg")
    img_stream2= return_img_stream("./Results/changemap.jpg")
    return   render_template('test.html',T1_Steam=T1_Steam,T2_Steam=T2_Steam,img_stream=img_stream,img_stream2=img_stream2)
  
def get_current_memory_gb() -> int:
    # 获取当前进程内存占用。
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    return info.uss / 1024. / 1024. 

def find_vector_set(diff_image, new_size):
   
    i = 0
    j = 0
    vector_set = np.zeros((int(new_size[0] * new_size[1] / 75), 75))

    print('\nvector_set shape',vector_set.shape)
    print("5-0.1--find_vector_set内存使用" , get_current_memory_gb(),'MB')
    while i < vector_set.shape[0]:
        print("5--find_vector_set内存使用" ,i, get_current_memory_gb(),'MB')
        while j < new_size[0]:
            print("5--find_vector_set内存使用" ,i, '---',j,get_current_memory_gb(),'MB')
            k = 0
            while k < new_size[1]:
                block   = diff_image[j:j+5, k:k+5]
                #print(i,j,k,block.shape)
                feature = block.ravel()
                vector_set[i, :] = feature
                k = k + 5
            j = j + 5
        i = i + 1
        
            
    mean_vec   = np.mean(vector_set, axis = 0)    
    vector_set = vector_set - mean_vec
    
    return vector_set, mean_vec
    
  
def find_FVS(EVS, diff_image, mean_vec, new):
    
    i = 2 
    feature_vector_set = []
    
    while i < new[0] - 2:
        j = 2
        print("6--find_FVS内存使用" , get_current_memory_gb())
        while j < new[1] - 2:
            block = diff_image[i-2:i+3, j-2:j+3]
            feature = block.flatten()
            feature_vector_set.append(feature)
            j = j+1
        i = i+1
        
    FVS = np.dot(feature_vector_set, EVS)
    FVS = FVS - mean_vec
    print("\nfeature vector space size",FVS.shape)
    return FVS

def clustering(FVS, components, new):
    
    kmeans = KMeans(components, verbose = 0)
    
    kmeans.fit(FVS)
    output = kmeans.predict(FVS)
    count  = Counter(output)

    least_index = min(count, key = count.get)            
    print(new[0],new[1])
    change_map  = np.reshape(output,(new[0] - 4, new[1] - 4))
    
    return least_index, change_map

   
def find_PCAKmeans(imagepath1, imagepath2):
    
    print('Operating')
    
    image1 = imread(imagepath1,pilmode="RGB")
    image2 = imread(imagepath2,pilmode="RGB")
    # print(image1.shape,image2.shape) 
    new_size = np.asarray(image1.shape) / 5
    new_size = new_size.astype(int) * 5
    image1 = imresize(image1, (new_size)).astype(np.int16)
    image2 = imresize(image2, (new_size)).astype(np.int16) 
    diff_image = abs(image1 - image2)   
    #imsave('diff.jpg', diff_image)
    imageio.imwrite('./Results/diff.jpg', diff_image)
    print('\nBoth images resized to ',new_size)
        
    vector_set, mean_vec = find_vector_set(diff_image, new_size)
    
    pca     = PCA()
    pca.fit(vector_set)
    EVS = pca.components_
        
    FVS     = find_FVS(EVS, diff_image, mean_vec, new_size)
    
    print('\ncomputing k means')
    
    components = 3
    least_index, change_map = clustering(FVS, components, new_size)
    
    change_map[change_map == least_index] = 255
    change_map[change_map != 255] = 0
    
    change_map = change_map.astype(np.uint8)
    kernel     = np.asarray(((0,0,1,0,0),
                             (0,1,1,1,0),
                             (1,1,1,1,1),
                             (0,1,1,1,0),
                             (0,0,1,0,0)), dtype=np.uint8)
    cleanChangeMap = cv2.erode(change_map,kernel)
    
    imageio.imwrite("./Results/changemap.jpg", change_map)
    imageio.imwrite("./Results/cleanchangemap.jpg", cleanChangeMap)
    #imsave("changemap.jpg", change_map)
    #imsave("cleanchangemap.jpg", cleanChangeMap)
def return_img_stream(img_local_path):
  """
  工具函数:
  获取本地图片流
  :param img_local_path:文件单张图片的本地绝对路径
  :return: 图片流
  """
  import base64
  import chardet
  img_stream = ''
  f = open(img_local_path,'rb')
  data = f.read()
  print(chardet.detect(data))
 
  with open(img_local_path, 'rb') as img_f:

    img_stream = img_f.read()
    img_stream = base64.b64encode(img_stream).decode()
  return img_stream
 


# Python已经取消scipy库中imread，imresize，imsave三个函数的使用,在文件中直接写入imresize函数源代码，
def imresize(arr, size, interp='bilinear', mode=None):
    im = Image.fromarray(arr, mode=mode) 
    ts = type(size)
    if np.issubdtype(ts, np.signedinteger):
        percent = size / 100.0
        size = tuple((np.array(im.size)*percent).astype(int))
    elif np.issubdtype(type(size), np.floating):
        size = tuple((np.array(im.size)*size).astype(int))
    else:
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp]) 
    return np.array(imnew)


def imread(name, flatten=False, mode=None):
    """
    Read an image from a file as an array.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    name : str or file object
        The file name or file object to be read.
    flatten : bool, optional
        If True, flattens the color layers into a single gray-scale layer.
    mode : str, optional
        Mode to convert image to, e.g. ``'RGB'``.  See the Notes for more
        details.
    Returns
    -------
    imread : ndarray
        The array obtained by reading the image.
    Notes
    -----
    `imread` uses the Python Imaging Library (PIL) to read an image.
    The following notes are from the PIL documentation.
    `mode` can be one of the following strings:
    * 'L' (8-bit pixels, black and white)
    * 'P' (8-bit pixels, mapped to any other mode using a color palette)
    * 'RGB' (3x8-bit pixels, true color)
    * 'RGBA' (4x8-bit pixels, true color with transparency mask)
    * 'CMYK' (4x8-bit pixels, color separation)
    * 'YCbCr' (3x8-bit pixels, color video format)
    * 'I' (32-bit signed integer pixels)
    * 'F' (32-bit floating point pixels)
    PIL also provides limited support for a few special modes, including
    'LA' ('L' with alpha), 'RGBX' (true color with padding) and 'RGBa'
    (true color with premultiplied alpha).
    When translating a color image to black and white (mode 'L', 'I' or
    'F'), the library uses the ITU-R 601-2 luma transform::
        L = R * 299/1000 + G * 587/1000 + B * 114/1000
    When `flatten` is True, the image is converted using mode 'F'.
    When `mode` is not None and `flatten` is True, the image is first
    converted according to `mode`, and the result is then flattened using
    mode 'F'.
    """
 
    im = Image.open(name)
    return fromimage(im, flatten=flatten, mode=mode)
def fromimage(im, flatten=False, mode=None):
    """
    Return a copy of a PIL image as a numpy array.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    im : PIL image
        Input image.
    flatten : bool
        If true, convert the output to grey-scale.
    mode : str, optional
        Mode to convert image to, e.g. ``'RGB'``.  See the Notes of the
        `imread` docstring for more details.
    Returns
    -------
    fromimage : ndarray
        The different colour bands/channels are stored in the
        third dimension, such that a grey-image is MxN, an
        RGB-image MxNx3 and an RGBA-image MxNx4.
    """
    if not Image.isImageType(im):
        raise TypeError("Input is not a PIL image.")
 
    if mode is not None:
        if mode != im.mode:
            im = im.convert(mode)
    elif im.mode == 'P':
        # Mode 'P' means there is an indexed "palette".  If we leave the mode
        # as 'P', then when we do `a = array(im)` below, `a` will be a 2-D
        # containing the indices into the palette, and not a 3-D array
        # containing the RGB or RGBA values.
        if 'transparency' in im.info:
            im = im.convert('RGBA')
        else:
            im = im.convert('RGB')
 
    if flatten:
        im = im.convert('F')
    elif im.mode == '1':
        # Workaround for crash in PIL. When im is 1-bit, the call array(im)
        # can cause a seg. fault, or generate garbage. See
        # https://github.com/scipy/scipy/issues/2138 and
        # https://github.com/python-pillow/Pillow/issues/350.
        #
        # This converts im from a 1-bit image to an 8-bit image.
        im = im.convert('L')
 
    a = array(im)
    return a
_errstr = "Mode is unknown or incompatible with input array shape."

# docker启动服务不会走main函数
if __name__ == '__main__':
    print("1--内存使用" , get_current_memory_gb())
    app.run()
 





 




