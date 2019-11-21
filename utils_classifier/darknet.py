'''Darknet in PyTorch.
As seen here https://github.com/fastai/fastai/blob/master/courses/dl2/cifar10-darknet.ipynb
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

seed=0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)
    

def conv_layer(ni, nf, ks=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=ks, bias=False, stride=stride, padding=ks//2),
        nn.BatchNorm2d(nf, momentum=0.01),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Dropout2d(p = 0.05))

class ResLayer(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.conv1=conv_layer(ni, ni//2, ks=1)
        self.conv2=conv_layer(ni//2, ni, ks=3)
        
    def forward(self, x): 
        return x.add(self.conv2(self.conv1(x)))

dropout = 0.5
class Darknet(nn.Module):
    def make_group_layer(self, ch_in, num_blocks, stride=1):
        return [conv_layer(ch_in, int(np.round(ch_in*2)),stride=stride)
               ] + [(ResLayer(int(np.round(ch_in*2)))) for i in range(num_blocks)]

    def __init__(self, num_blocks, num_classes, ni=1, nf=32):
        nf_initial = nf
        super().__init__()
        layers = [conv_layer(ni, nf, ks=3, stride=1)]
        for i,nb in enumerate(num_blocks):
            layers += self.make_group_layer(nf, nb, stride=2-(i==1))
            nf = int(np.round(nf*2))
        layers += [nn.AdaptiveAvgPool2d((1)), Flatten(), 
                   nn.Dropout(dropout), nn.Linear(int((2**len(num_blocks))*nf_initial), num_classes)]
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x): return self.layers(x)

class Darknet_dropout(nn.Module):
    def make_group_layer(self, ch_in, num_blocks, stride=1):
        return [conv_layer(ch_in, int(np.round(ch_in*2)),stride=stride)
               ] + [(ResLayer(int(np.round(ch_in*2)))) for i in range(num_blocks)]

    def __init__(self, num_blocks, num_classes, dropout, nf=32):
        nf_initial = nf
        super().__init__()
        layers = [conv_layer(1, nf, ks=3, stride=1)]
        for i,nb in enumerate(num_blocks):
            layers += self.make_group_layer(nf, nb, stride=2-(i==1))
            nf = int(np.round(nf*2))
        layers += [nn.AdaptiveAvgPool2d((1)), Flatten(), 
                   nn.Dropout(dropout), nn.Linear(int((2**len(num_blocks))*nf_initial), num_classes)]
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x): return self.layers(x)

def conv_layer3D(ni, nf, ks=3, stride=1):
    return nn.Sequential(
        nn.Conv3d(ni, nf, kernel_size=ks, bias=False, stride=stride, padding=ks//2),
        nn.BatchNorm3d(nf, momentum=0.01),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Dropout3d(p = 0.05))

class ResLayer3D(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.conv1=conv_layer3D(ni, ni//2, ks=1)
        self.conv2=conv_layer3D(ni//2, ni, ks=3)
        
    def forward(self, x): 
        return x.add(self.conv2(self.conv1(x)))

class Darknet3D_dropout(nn.Module):
    def make_group_layer(self, ch_in, num_blocks, stride=1):
        return [conv_layer3D(ch_in, int(np.round(ch_in*2)),stride=stride)
               ] + [(ResLayer3D(int(np.round(ch_in*2)))) for i in range(num_blocks)]

    def __init__(self, num_blocks, num_classes, dropout, nf=32):
        nf_initial = nf
        super().__init__()
        layers = [conv_layer3D(1, nf, ks=3, stride=1)]
        for i,nb in enumerate(num_blocks):
            layers += self.make_group_layer(nf, nb, stride=2-(i==1))
            nf = int(np.round(nf*2))
        layers += [nn.AdaptiveAvgPool3d((1)), Flatten(), 
                   nn.Dropout(dropout), nn.Linear(int((2**len(num_blocks))*nf_initial), num_classes)]
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x): return self.layers(x)