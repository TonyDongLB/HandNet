from torch import nn
import torch as t
import torch
import math
import time

from model.modules import *
from torchvision.models import ResNet
from collections import OrderedDict


class CamNet(nn.Module):
    """
    基于SE_ResNet50结构，去掉的maxpool和全连接层，增加了fusion模块。
    注意在特征提取阶段，对img的特征提取反向传播两次，注意学习率。
    """
    def __init__(self, block=SEBottleneck, layers=(3, 4, 6, 3), pretrained=False):
        super(CamNet, self).__init__()

        self.img_inplanes = 64
        self.flow_inplanes = 64

        self.img_feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(block, 64, layers[0], img_path=True),
            self._make_layer(block, 128, layers[1], stride=2, img_path=True),
            self._make_layer(block, 256, layers[2], stride=2, img_path=True),
        )

        self.flow_feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(block, 64, layers[0], img_path=False),
            self._make_layer(block, 128, layers[1], stride=2, img_path=False),
            self._make_layer(block, 256, layers[2], stride=2, img_path=False),
        )

        self.fusion4img = nn.Sequential(
            nn.Conv2d(256 * block.expansion * 2, 256 * block.expansion, kernel_size=1, bias=False),
            nn.Conv2d(256 * block.expansion, 256 * block.expansion, kernel_size=3, padding=1, bias=False),
        )
        self.fusion4flow = nn.Sequential(
            nn.Conv2d(256 * block.expansion * 2, 256 * block.expansion, kernel_size=1, bias=False),
            nn.Conv2d(256 * block.expansion, 256 * block.expansion, kernel_size=3, padding=1, bias=False),
        )
        self.layer4fusion = self._make_layer(block, 512, layers[3], stride=2, img_path=True)
        self.layer4flow = self._make_layer(block, 512, layers[3], stride=2, img_path=False)

        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.fc_fusion = nn.Sequential(
            nn.Linear(512 * block.expansion, 512 * block.expansion // 2),
            nn.Linear(512 * block.expansion // 2, 2),
        )
        self.fc_flow = nn.Sequential(
            nn.Linear(512 * block.expansion, 512 * block.expansion // 2),
            nn.Linear(512 * block.expansion // 2, 2),
        )

        # # 初始化
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for mm in m:
                    if isinstance(mm, nn.Conv2d):
                        n = mm.kernel_size[0] * mm.kernel_size[1] * mm.out_channels
                        mm.weight.data.normal_(0, math.sqrt(2. / n))
                    elif isinstance(mm, nn.BatchNorm2d):
                        mm.weight.data.fill_(1)
                        mm.bias.data.zero_()
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, img_path=True):
        downsample = None
        if img_path:
            inplanes = self.img_inplanes
        else:
            inplanes = self.flow_inplanes
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        if img_path:
            inplanes = planes * block.expansion
            self.img_inplanes = inplanes
        else:
            inplanes = planes * block.expansion
            self.flow_inplanes = inplanes
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, img1, img2, flow):
        img1_fea = self.img_feature(img1)
        img2_fea = self.img_feature(img2)
        flow_fea = self.flow_feature(flow)
        img_fea = torch.cat((img1_fea, img2_fea), dim=1)
        img_fea = self.fusion4img(img_fea)
        fusioned_fea = torch.cat((img_fea, flow_fea), dim=1)
        fusioned_fea = self.fusion4flow(fusioned_fea)
        fusioned_fea = self.layer4fusion(fusioned_fea)
        flow_fea = self.layer4flow(flow_fea)

        fusioned_fea = self.avgpool(fusioned_fea)
        fusioned_fea = fusioned_fea.view(fusioned_fea.size(0), -1)
        flow_fea = self.avgpool(flow_fea)
        flow_fea = flow_fea.view(flow_fea.size(0), -1)

        result1 = self.fc_fusion(fusioned_fea)
        result2 = self.fc_flow(flow_fea)

        return result1.div(2) + result2.div(2)

    def save(self, name=None, epoch=0, batch_size=128, eval=0):
        '''
        保存模型，默认使用“模型名字+种类数+时间”作为文件名
        '''
        if name is None:
            prefix = 'checkpoints/epoch_' + str(epoch + 1) + '_' + str(eval) + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name

    def load(self, path, gpu=False):
        '''
        可加载指定路径的模型，针对是在多GPU训练的模型。
        '''
        if gpu:
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location='cpu')

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # print(k)
            name = k#[7:]  # remove `module.`
            new_state_dict[name] = v
        self.load_state_dict(new_state_dict)

if __name__ == "__main__":
    test = CamNet()
    dic = test.state_dict()
    print(len(dic))
    print(list(dic.keys()))