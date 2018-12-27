import os
from torch.utils import data
import numpy as np
import torch
from torchvision import transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader
import glob
import cv2

class Hand(data.Dataset):
    # # to load the hand picture
    def __init__(self, root, transforms=None, train=False, test=False, deploy=False):
        super(Hand, self).__init__()
        self.root = root
        self.transforms = transforms
        self.train = train
        self.test = test
        self.deploy = deploy

        if test:
            self.imgs = glob.glob(root + '/test/0' + "/*." + 'jpg')
            self.imgs += glob.glob(root + '/test/1' + "/*." + 'jpg')
        elif train:
            self.imgs = glob.glob(root + '/train/0' + "/*." + 'jpg')
            self.imgs += glob.glob(root + '/train/1' + "/*." + 'jpg')
        else:
            self.imgs = glob.glob(root + '/deploy' + "/*." + 'jpg')
        if transforms is None:
            # normalize need to edit!
            normalize = T.Normalize(mean=[0.4, 0.4, 0.4], std=[0.4, 0.4, 0.4])
            self.transforms4img = T.Compose([
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        filename = img_path.split('/')[-1].split('.')[0]
        num = filename.split("_")[-1]
        next_num = str(int(num) + 10)
        before_num = str(int(num) - 10)
        next_img_path = img_path.replace(num, next_num)
        before_img_path = img_path.replace(num, before_num)
        if not os.path.exists(next_img_path):
            next_img_path = img_path
        if not os.path.exists(before_img_path):
            before_img_path = img_path

        img_0 = cv2.imread(before_img_path)
        img_1 = cv2.imread(img_path)
        img_2 = cv2.imread(next_img_path)
        img_0 = cv2.resize(img_0, (224, 224))
        img_1 = cv2.resize(img_1, (224, 224))
        img_2 = cv2.resize(img_2, (224, 224))
        flow0, flow1 = None, None
        flow0 = cv2.calcOpticalFlowFarneback(cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY),
                                            cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY),
                                            flow=flow1,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2,
                                            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        flow1 = cv2.calcOpticalFlowFarneback(cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY),
                                            cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY),
                                            flow=flow1,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2,
                                            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        attach0 = np.zeros((flow0.shape[0], flow0.shape[1], 1), flow0.dtype)
        attach1 = np.zeros((flow1.shape[0], flow1.shape[1], 1), flow1.dtype)
        flow0 = np.clip(flow0, -40, 40)
        flow1 = np.clip(flow1, -40, 40)
        flow0 += 40
        flow1 += 40
        flow0 /= 80
        flow1 /= 80
        flow0 = np.concatenate((flow0, attach0), 2)
        flow1 = np.concatenate((flow1, attach1), 2)

        # # add a random flip
        if np.random.random() > 0.5 and not self.deploy:
            img_1 = np.flip(img_1, 1).copy()
            flow0 = np.flip(flow0, 1).copy()
            flow1 = np.flip(flow1, 1).copy()


        img_1 = self.transforms4img(img_1)
        flow0 = np.transpose(flow0, (2, 0, 1))
        flow1 = np.transpose(flow1, (2, 0, 1))

        if self.deploy:
            if next_img_path == img_path:
                label = '-1'
            else:
                label = filename
        else:
            if img_path == next_img_path or img_path == before_img_path:
                label = 0
            else:
                label = int(img_path.split('/')[-2])

        return img_1, flow0, flow1, label

    def __len__(self):
        return len(self.imgs)

