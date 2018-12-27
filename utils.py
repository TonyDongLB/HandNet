import torch as t
from torchvision.models import resnet50
from model import CamNet
from collections import OrderedDict
from model.modules import *

def loadPretrain(net):
    print('load pretrained model...')
    resnet = resnet50(pretrained=True)
    res_dict = resnet.state_dict()
    model = net.state_dict()
    new_dict = {}

    for (model_key, model.val), (res_key, res_val) in zip(model.items(), res_dict.items()):
        if model_key.startswith('img'):
            new_dict[model_key] = res_val
        else:
            break

    for model_key, model.val in model.items():
        if model_key.startswith('flow'):
            now_key = model_key.replace('flow', 'img')
            new_dict[model_key] = new_dict[now_key]

    # for (model_key, model.val), (res_key, res_val) in zip(model.items(), res_dict.items()):
    #     print((model_key) + '       ' + (res_key))
    #     if model_key.find('flow') >= 0:
    #         print(model_key)
    #         new_dict[model_key] = res_val

    model.update(new_dict)
    net.load_state_dict(model)
    return net
