import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, n_in, n_out, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.full = nn.Sequential(
            conv3x3(n_in, n_out, stride),
            nn.BatchNorm2d(n_out),
            nn.ReLU(inplace=True),
            conv3x3(n_out, n_out),
            nn.BatchNorm2d(n_out)
        )
        self.skip = conv1x1(n_in, n_out)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.full(x)

        if self.downsample is not None:
            identity = self.downsample(x)
        identity = self.skip(identity)
        out += identity
        out = F.relu(out)

        return out

class Hourglass(nn.Module):
    def __init__(self, n_in, n_out, n):
        super(Hourglass, self).__init__()
        self.upper_branch = nn.Sequential(
            ResBlock(n_in, 256),
            ResBlock(256, 256),
            ResBlock(256, n_out)
        )
        
        self.lower_branch = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            ResBlock(n_in, 256),
            ResBlock(256, 256),
            ResBlock(256, 256)
        )
        
        if n > 1:
            self.inter = Hourglass(256, n_out, n-1)
        else:
            self.inter = ResBlock(256, n_out)
        self.last = ResBlock(n_out, n_out)
        
    def forward(self, x):
        upper = self.upper_branch(x)
        lower = self.lower_branch(x)
        lower = self.inter(lower)
        lower = self.last(lower)
        lower = F.interpolate(lower, scale_factor=2)
        return upper + lower