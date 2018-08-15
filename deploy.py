import sys
import os
from optparse import OptionParser
import numpy as np
import glob
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import cv2
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from PIL import Image
from torchvision import transforms as T
from model import CamNet
from dataset import Hand
from loss import FocalLoss2d

os.environ["CUDA_VISIBLE_DEVICES"]="0"

normalize = T.Normalize(mean=[0.4, 0.4, 0.4], std=[0.4, 0.4, 0.4])
transforms4img = T.Compose([
                T.ToTensor(),
                normalize
            ])


def get_tensors_from_path(img_path):
    filename = img_path.split('/')[-1].split('.')[0]
    num = filename.split("_")[-1]
    next_num = str(int(num) + 10)
    next_img_path = img_path.replace(num, next_num)
    # 不存在下一个帧
    if not os.path.exists(next_img_path):
        return None
    img_1 = cv2.imread(img_path)
    img_2 = cv2.imread(next_img_path)
    img_1 = cv2.resize(img_1, (224, 224))
    img_2 = cv2.resize(img_2, (224, 224))
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
    img_1 = transforms4img(img_1)
    img_2 = transforms4img(img_2)
    flow = np.transpose(flow, (2, 0, 1))
    flow = torch.from_numpy(flow)
    label = filename
    img_1 = torch.unsqueeze(img_1, dim=0)
    img_2 = torch.unsqueeze(img_2, dim=0)
    flow = torch.unsqueeze(flow, dim=0)
    return img_1, img_2, flow, label


def deploy(model, gpu):
    root_data = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'data')
    # deploy_set = Hand(root_data, deploy=True)
    # deploy_data = DataLoader(deploy_set, batch_size=1, shuffle=False,
    #                        num_workers=4)
    imgs = glob.glob(root_data + '/deploy' + "/*." + 'jpg')

    for img_path in imgs:
        result = get_tensors_from_path(img_path)
        if result is None:
            img = cv2.imread(img_path)
            filename = img_path.split('/')[-1].split('.')[0]
            cv2.imwrite('data/deploy/0/' + filename + '_1.jpg', img)
            continue
        else:
            img1, img2, flow, label = result

        img1 = Variable(img1)
        img2 = Variable(img2)
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
        if classed == 0:
            if cv2.imwrite('data/deploy/0/' + label + '_1.jpg', img1):
                print('Writed image {}'.format(label))
            else:
                print('Something wrong {}'.format('data/deploy/0/' + label[0] + '_1.jpg'))
        else:
            if cv2.imwrite('data/deploy/1/' + label + '_1.jpg', img1):
                print('Writed image {}'.format(label))
            else:
                print('Something wrong')


def get_args():
    parser = OptionParser()
    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,
                      type='int', help='batch size')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('--model', '-m', default='checkpoint/good_perform.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    model = CamNet()


    if args.gpu:
        model.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory
        print("Using ", torch.cuda.device_count(), " GPUs!")
    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), " GPUs!")
        model = nn.DataParallel(model)

    model.load(path=args.model, gpu=args.gpu)
    model.eval()
    print('Processing!')
    deploy(model=model, gpu=args.gpu)
    print('DONE')

