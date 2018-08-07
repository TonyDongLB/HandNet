import os
from torch.utils import data
import numpy as np
import torch
from torchvision import transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader
import glob
import cv2
from torch import nn
import torch as t
import torch
import math
from model import CamNet

from torchvision.models import ResNet
from collections import OrderedDict

from matplotlib import pyplot as plt

# 参数依次为list,抬头,X轴标签,Y轴标签,XY轴的范围
def draw_hist(myList,Title,Xlabel,Ylabel,Xmin,Xmax,Ymin,Ymax):
    plt.hist(myList,100)
    plt.xlabel(Xlabel)
    plt.xlim(Xmin,Xmax)
    plt.ylabel(Ylabel)
    plt.ylim(Ymin,Ymax)
    plt.title(Title)
    plt.show()

# # 查看权重文件
# pkl_path = 'model/weight-99.pkl'
# weight_ori = torch.load(pkl_path, map_location='cpu')
# weight_ori = weight_ori['weight']
# weight_new = OrderedDict
# print(len(weight_ori))
# for key in list(weight_ori.keys()):
#     print(key)

# # 查看光流
imgs = glob.glob('/Users/apple/Documents/NN_Models/HandNet/data' + '/train/0' + "/*." + 'jpg')
imgs += glob.glob('/Users/apple/Documents/NN_Models/HandNet/data' + '/train/1' + "/*." + 'jpg')
print(len(imgs))
MAX = []
MIN =[]
for index in range(len(imgs)):
    img_path = imgs[index]
    # print(img_path)
    filename = img_path.split('/')[-1].split('.')[0]
    num = filename.split('_')[-1]
    next_num = str(int(num) + 10)
    next_img_path = img_path.replace(num, next_num)
    if not os.path.exists(next_img_path):
        next_img_path = img_path

    img_1 = cv2.imread(img_path)
    img_2 = cv2.imread(next_img_path)
    img_1 = cv2.resize(img_1, (224, 224))
    img_2 = cv2.resize(img_2, (224, 224))

    flow = None
    flow = cv2.calcOpticalFlowFarneback(prev=cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY),
                                        next=cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY),
                                        flow=flow,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2,
                                        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    attach = np.zeros((flow.shape[0], flow.shape[1], 1), flow.dtype)
    flow = np.concatenate((flow, attach), 2)
    print(flow.shape)
    print(flow.dtype)

draw_hist(MAX, 'pos', 'i', 'num', np.min(MAX), 400, 0, 200)
draw_hist(MIN, 'neg', 'i', 'num', np.min(MIN), 400, 0, 200)
#
# print(np.mean(all))
# print(np.max(all))

# model = CamNet()
# model = nn.DataParallel(model)
# print(model)