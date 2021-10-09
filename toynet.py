import torch
from torch import nn,Tensor
from torch.nn import functional as F


class BatchNorm(nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        super(BatchNorm,self).__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)

    def _check_input_dim(self, input:Tensor)->Tensor:
        pass


class GlobalAvgPool2d(nn.Module):
    def __init__(self,keepdim:bool=False)->None:
        super(GlobalAvgPool2d,self).__init__()
        self.keepdim=keepdim

    def forward(self,input:Tensor)->Tensor:
        return torch.mean(input,[2,3],self.keepdim)


class ToyNet(nn.Module):
    def __init__(self,input_shape:list[int,int,int]=[1,28,28],classes:int=10)->None:
        super(ToyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 6, 3, 1, 1)
        self.conv2 = nn.Conv2d(6, 18, 3, 2, 1)
        self.conv3 = nn.Conv2d(18, 32, 3, 2, 1)
        self.conv4 = nn.Conv2d(32, 48, 3, 1, 1)
        self.conv5 = nn.Conv2d(48, 96, 3, 1, 1)
        self.global_pool = GlobalAvgPool2d()
        self.linear = nn.Linear(96, classes)

    def forward(self, x:Tensor)->Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.global_pool(x)
        x = self.linear(x)
        return x


class ToyNet_BN(nn.Module):
    def __init__(self,input_shape:list[int,int,int]=[1,28,28],classes:int=10)->None:
        super(ToyNet_BN, self).__init__()
        self.bn0 = BatchNorm(input_shape[0])
        self.conv1 = nn.Conv2d(input_shape[0], 6, 3, 1, 1,bias=False)
        self.bn1 = BatchNorm(6)
        self.conv2 = nn.Conv2d(6, 18, 3, 2, 1,bias=False)
        self.bn2 = BatchNorm(18)
        self.conv3 = nn.Conv2d(18, 32, 3, 2, 1,bias=False)
        self.bn3 = BatchNorm(32)
        self.conv4 = nn.Conv2d(32, 48, 3, 1, 1,bias=False)
        self.bn4 = BatchNorm(48)
        self.conv5 = nn.Conv2d(48, 96, 3, 1, 1,bias=False)
        self.bn5 = BatchNorm(96)
        self.global_pool = GlobalAvgPool2d()
        self.linear = nn.Linear(96, classes)

    def forward(self, x:Tensor)->Tensor:
        x = self.bn0(x)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.global_pool(x)
        x = self.linear(x)
        return x