import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


'''
PIP_pde
'''

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# ----------------  pde ---------------


class HFunction(nn.Module):
    def __init__(self):
        super(HFunction, self).__init__()
        self.conv = nn.Sequential(           
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1)
        )


    def forward(self, f_x1, f_x2):


        return self.conv(torch.cat([f_x1,f_x2],dim=1))


class pde(nn.Module):
    def __init__(self):
        super(pde, self).__init__()

        self.d02=Conv_d02()
        self.d04=Conv_d04()
        self.d06=Conv_d06()
        self.d08=Conv_d08()

        self.d13=Conv_d13()
        self.d84=Conv_d84()
        self.d75=Conv_d75()

        self.d53=Conv_d53()
        self.d62=Conv_d62()
        self.d71=Conv_d71()


        self.g_x = HFunction()

    def forward(self,x, f_x1): # f_x1是步长，f_x2是控制参数
        _, _, hei, wid = x.shape

        d02 = self.d02(x)
        d04 = self.d04(x)
        d06 = self.d06(x)
        d08 = self.d08(x)

        d13 = self.d13(x)
        d84 = self.d84(x)
        d75 = self.d75(x)

        d53 = self.d53(x)
        d62 = self.d62(x)
        d71 = self.d71(x)

        c02 = 1 / (1e-8 + torch.sqrt(torch.square(d02) + (torch.square(d13) + torch.square(d84)) / 8))
        c06 = 1 / (1e-8 + torch.sqrt(torch.square(d06) + (torch.square(d75) + torch.square(d84)) / 8))
        c04 = 1 / (1e-8 + torch.sqrt(torch.square(d04) + (torch.square(d53) + torch.square(d62)) / 8))
        c08 = 1 / (1e-8 + torch.sqrt(torch.square(d08) + (torch.square(d71) + torch.square(d62)) / 8))

        hamilton = self.g_x(x,f_x1)
        nabla = c02 * torch.abs(d02) + c04 * torch.abs(d04) + c06 * torch.abs(d06) + c08 * torch.abs(d08)
        delta_x = x + nabla + hamilton

        return torch.sigmoid(delta_x)

# ----------------  Directional Conv2d *n ---------------


class Conv_d01(nn.Module):
    def __init__(self):
        super(Conv_d01, self).__init__()
        kernel = [[1, 0, 0],
                  [0, -1, 0],
                  [0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)


class Conv_d02(nn.Module):
    def __init__(self):
        super(Conv_d02, self).__init__()
        kernel = [[0, 1, 0],
                  [0, -1, 0],
                  [0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)


class Conv_d03(nn.Module):
    def __init__(self):
        super(Conv_d03, self).__init__()
        kernel = [[0, 0, 1],
                  [0, -1, 0],
                  [0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)


class Conv_d04(nn.Module):
    def __init__(self):
        super(Conv_d04, self).__init__()
        kernel = [[0, 0, 0],
                  [0, -1, 1],
                  [0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)


class Conv_d05(nn.Module):
    def __init__(self):
        super(Conv_d05, self).__init__()
        kernel = [[0, 0, 0],
                  [0, -1, 0],
                  [0, 0, 1]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)


class Conv_d06(nn.Module):
    def __init__(self):
        super(Conv_d06, self).__init__()
        kernel = [[0, 0, 0],
                  [0, -1, 0],
                  [0, 1, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)


class Conv_d07(nn.Module):
    def __init__(self):
        super(Conv_d07, self).__init__()
        kernel = [[0, 0, 0],
                  [0, -1, 0],
                  [1, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)


class Conv_d08(nn.Module):
    def __init__(self):
        super(Conv_d08, self).__init__()
        kernel = [[0, 0, 0],
                  [1, -1, 0],
                  [0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)

class Conv_d53(nn.Module):
    def __init__(self):
        super(Conv_d53, self).__init__()
        kernel = [[0, 0, 1],
                  [0, 0, 0],
                  [0, 0, -1]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)

class Conv_d62(nn.Module):
    def __init__(self):
        super(Conv_d62, self).__init__()
        kernel = [[0, 1, 0],
                  [0, 0, 0],
                  [0, -1, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)

class Conv_d71(nn.Module):
    def __init__(self):
        super(Conv_d71, self).__init__()
        kernel = [[1, 0, 0],
                  [0, 0, 0],
                  [-1, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)


class Conv_d13(nn.Module):
    def __init__(self):
        super(Conv_d13, self).__init__()
        kernel = [[-1, 0, 1],
                  [0, 0, 0],
                  [0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)

class Conv_d84(nn.Module):
    def __init__(self):
        super(Conv_d84, self).__init__()
        kernel = [[0, 0, 0],
                  [-1, 0, 1],
                  [0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)

class Conv_d75(nn.Module):
    def __init__(self):
        super(Conv_d75, self).__init__()
        kernel = [[0, 0, 0],
                  [0, 0, 0],
                  [-1, 0, 1]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)

# if __name__ == '__main__':
#     net = TFD(4,4)
#     x = torch.randn(1,4,3,3)
#     y = torch.randn(1,1,3,3)
#     z = net(x,y)
#     print(z)