####################################################
##### This is focal loss class for multi class #####
##### University of Tokyo Doi Kento            #####
####################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# I referred https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

# modified by Dongli and the original edition is piece of shit

class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2, weight=None, size_average=True):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, data, target):
        if data.dim() == 4:
            data = data.contiguous().view(data.size(0), data.size(1), -1)
            data = data.transpose(1, 2)
            data = data.contiguous().view(-1, data.size(2)).squeeze()

        if target.dim() == 4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim() == 3:
            target = target.view(-1)


        # compute the negative likelyhood
        if self.weight is None:
            logpt = -F.cross_entropy(data, target)
        else:
            weight = Variable(self.weight)
            logpt = -F.cross_entropy(data, target, weight=weight)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1-pt)**self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()




if __name__ == "__main__":
    FL = FocalLoss2d(gamma=0)
    CE = nn.CrossEntropyLoss()
    N = 5
    C = 4
    inputs = torch.rand(1, 2, N, C)
    targets = torch.LongTensor(1, 1, N, C,).random_(2)
    inputs_fl = Variable(inputs.clone(), requires_grad=True)
    targets_fl = Variable(targets.clone())

    inputs_ce = Variable(inputs.clone(), requires_grad=True)
    targets_ce = Variable(targets.clone())
    print('----inputs----')
    print(inputs_fl)
    print('---target-----')
    print(targets_fl)
    # print('----trans----')
    # trans = inputs_fl.transpose(0, 2)
    # trans = trans.contiguous().view(-1, trans.size(2)).squeeze().shape
    # print(trans)

    fl_loss = FL(inputs_fl, targets_fl)
    # ce_loss = CE(inputs_ce, targets_ce)
    # print('ce = {}'.format(ce_loss.data[0]))
    print('fl ={}'.format(fl_loss.data[0]))
    fl_loss.backward()
    # ce_loss.backward()
    print(inputs_fl.grad.data)
    # print(inputs_ce.grad.data)