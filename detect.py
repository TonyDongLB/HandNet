import numpy as np
import cv2
import threading

from torchvision import transforms as T
from model import CamNet
from torch.autograd import Variable
import torch.nn as nn
import torch
import os
from optparse import OptionParser


class DLtoPredict(object):
    def __init__(self, model, gpu=False):
        super(DLtoPredict, self).__init__()
        self.model = model
        self.gpu = gpu
        normalize = T.Normalize(mean=[0.4, 0.4, 0.4], std=[0.4, 0.4, 0.4])
        self.transforms4img = T.Compose([
            T.ToTensor(),
            normalize
        ])

    def get_tensor_from_numpy(self, img_1, img_2):
        flow = None
        flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY),
                                            cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY),
                                            flow=flow,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2,
                                            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        attach = np.zeros((flow.shape[0], flow.shape[1], 1), flow.dtype)
        flow = np.clip(flow, -20, 20)
        flow += 20
        flow /= 40
        flow = np.concatenate((flow, attach), 2)
        img_1 = self.transforms4img(img_1)
        img_2 = self.transforms4img(img_2)
        flow = np.transpose(flow, (2, 0, 1))
        flow = torch.from_numpy(flow)
        img_1 = torch.unsqueeze(img_1, dim=0)
        img_2 = torch.unsqueeze(img_2, dim=0)
        flow = torch.unsqueeze(flow, dim=0)
        print('have done to tensor')
        return img_1, img_2, flow

    def predict(self, img_1, img_2, flow):
        model = self.model
        gpu = self.gpu
        img1 = Variable(img_1)
        img2 = Variable(img_2)
        flow = Variable(flow)

        if gpu:
            img1 = img1.cuda()
            img2 = img2.cuda()
            flow = flow.cuda()

        output = model(img1, img2, flow)
        pred = output.data.max(1)[1]
        classed = int(pred.cpu().numpy()[0])
        img1 = img1.cpu().data.squeeze(dim=0).numpy()
        img2 = img2.cpu().data.squeeze(dim=0).numpy()
        img1 = ((img1 * 0.4) + 0.4) * 255
        img2 = ((img2 * 0.4) + 0.4) * 255
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)
        if classed == 1:
            if cv2.imwrite('data/deploy/0/' + label + '_1.jpg', img1):
                print('Writed image {}'.format(label))
            else:
                print('Something wrong {}'.format('data/deploy/0/' + label[0] + '_1.jpg'))
        else:
            if cv2.imwrite('data/deploy/1/' + label + '_1.jpg', img1):
                print('Writed image {}'.format(label))
            else:
                print('Something wrong')


class FramePutter(threading.Thread):
    def __init__(self, producer):
        super(FramePutter, self).__init__()
        self.producer = producer

    def run(self):
        while event_not_signed_frame.is_set():
            threadLock.acquire()
            image = self.producer()
            frames.append(image)
            threadLock.release()


class FrameGetter(threading.Thread):
    def __init__(self):
        super(FrameGetter, self).__init__()

    def run(self):
        while event_not_signed_frame.is_set():
            if len(frames) > 1 and len(to_process) == 0:
                threadLock.acquire()
                to_process.append(frames[0])
                to_process.append(frames[1])
                del frames[0], frames[1]
                threadLock.release()


class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
        # 对于 python2.7 或者低版本的 numpy 请使用 jpeg.tostring()
        return image


threadLock = threading.Lock()
frames = []
to_process = []
event_not_signed_frame = threading.Event()  # 如果出现签字的帧，则激活此event

if __name__ == '__main':
    if not event_not_signed_frame.is_set():
        event_not_signed_frame.set()
    frames = []

    pass
