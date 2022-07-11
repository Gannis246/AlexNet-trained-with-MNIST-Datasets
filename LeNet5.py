# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 01:47:45 2022

@author: lenovo
"""

import torch
import torch.nn as nn
import numpy as np

'''
LeNet-5主干网络
'''
class LeNet5(nn.Module): # 用LeNet-5继承基类nn.Module
    def __init__(self):
        super(LeNet5, self).__init__() # 将父类的对象传给LeNet-5
        
        self.C1 = nn.Conv2d(1, 6, 5, padding = 2, padding_mode = 'replicate') # C1层：输入通道1，输出通道6，kernel_size=5
        self.S2 = Subsampling(6) # 池化层
        self.C3 = MapConv(6, 16, 5) # C3层：映射卷积层
        self.S4 = Subsampling(16) # 池化层
        self.C5 = nn.Conv2d(16, 120, 5)
        self.F6 = nn.Linear(120, 84)
        self.Output = RBFLayer(84, 10, RBF_WEIGHT)
        
        self.act = nn.Tanh() # 非线性激活
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                F_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data = torch.rand(m.weight.data.size()) * 4.8 / F_in - 2.4 / F_in
            
            elif isinstance(m, nn.Conv2d):
                F_in = m.in_features
                m.weight.data = torch.rand(m.weight.data.size()) * 4.8 / F_in - 2.4 / F_in
        
    def forward(self, x):
        x = self.C1(x)
        x = 1.7159 * self.act(2*self.S2(x)/3)
        x = self.C3(x)
        x = 1.7159 * self.act(2*self.S4(x)/3)
        x = self.C5(x)
        
        x = x.view(-1, 120)
        
        x = 1.7159 * self.act(2*self.F6(x)/3)
        
        out = self.Output(x)
        return out
    
    
'''
S2:采样层（池化层）
采用特殊平均池化 y=w(a1+a2+a3+a4)+b，其中w，b为可训练参数
in_channel:输入通道（特征图）数
'''   
class Subsampling(nn.Module): # 直接继承nn.Module类，方便进行计算和自动求导
    def __init__(self, in_channel):
        super(Subsampling, self).__init__()
        
        self.pool = nn.AvgPool2d(2) # 平均池化，相当于直接求(a1+a2+a3+a4)
        self.in_channel = in_channel
        F_in = 4 * self.in_channel # 4:池化区域为2x2，F_in:所有通道池化参数个数
        # 将weight和bias定义为nn.Parameter，方便求导。
        # weight的范围为[-2.4 / F_in, 2.4 / F_in)
        # bias的范围为[0,1)
        self.weight = nn.Parameter(torch.rand(self.in_channel) * 4.8 / F_in - 2.4 / F_in, requires_grad=True)
        self.bias = nn.Parameter(torch.rand(self.in_channel), requires_grad=True)
        
    def forward(self, x):
        x = self.pool(x) # 求出每个2x2小图的平均数
        outs = [] # 对每个channel的特征图进行池化，结果储存在这里
        
        for channel in range(self.in_channel):
            out = x[:, channel] * self.weight[channel] + self.bias[channel] # 计算每个channel的池化结果[batch_size, height, weight]
            outs.append(out.unsqueeze(1)) # 升维为[channel, batch_size, height, weight]
        return torch.cat(outs, dim = 1) # 将维度恢复为[batch_size, channel, height, weight]


'''
C3:卷积层
每个输出的特征图只挑选一小部分进行卷积
映射方式如：0号输出特征图由0，1，2号输入特征图卷积得到
'''    
class MapConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = 5):
        super(MapConv, self).__init__()
        # 定义特征图的映射方式
        # 纵坐标为6个输入特征图通道，横坐标为16个输出特征图通道
        mapInfo = [[1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1],
                   [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                   [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1],
                   [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
                   [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
                   [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1]]
        mapInfo = torch.tensor(mapInfo, dtype = torch.long) # 将映射矩阵转换为long型Tensor
        self.register_buffer("mapInfo", mapInfo) # buffer不会被求梯度
        
        self.in_channel = in_channel # 输入通道数
        self.out_channel = out_channel # 输出通道数（卷积核数）
        
        self.convs = {} # 用于放置每个定义的卷积层
        
        for i in range(self.out_channel):
            conv = nn.Conv2d(mapInfo[:, i].sum().item(), 1, kernel_size) # 定义每个单独的卷积核，输入通道数为映射矩阵列求和
            convName = "conv{}".format(i) # 命名
            self.convs[convName] = conv
            self.add_module(convName, conv)
         
    def forward(self, x):
        outs = [] # 将映射卷积的结果放置于此
        
        for i in range(self.out_channel):
            mapIdx = self.mapInfo[:, i].nonzero().squeeze() # 求每列为1的位置索引并压缩为一维
            convInput = x.index_select(1, mapIdx) # 选择映射 index_select(dim, index)
            convOutput = self.convs['conv{}'.format(i)](convInput) #卷积计算
            outs.append(convOutput)
        return torch.cat(outs, dim=1) # 在一维上连接
        
    
'''
输出层：
提取像素编码，求欧式（不加根号，平方和）距离进行特征匹配
'''
_zero = [-1, +1, +1, +1, +1, +1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, -1, +1, +1, +1, -1, -1] + \
        [-1, +1, +1, -1, +1, +1, -1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [-1, +1, +1, -1, +1, +1, -1] + \
        [-1, -1, +1, +1, +1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1]

_one = [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, +1, +1, +1, -1, -1] + \
       [-1, +1, +1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, +1, +1, +1, +1, +1, +1] + \
       [-1, -1, -1, -1, -1, -1, -1] + \
       [-1, -1, -1, -1, -1, -1, -1]

_two = [-1, +1, +1, +1, +1, +1, -1] + \
       [-1, -1, -1, -1, -1, -1, -1] + \
       [-1, +1, +1, +1, +1, +1, -1] + \
       [+1, +1, -1, -1, -1, +1, +1] + \
       [+1, -1, -1, -1, -1, +1, +1] + \
       [-1, -1, -1, -1, +1, +1, -1] + \
       [-1, -1, +1, +1, +1, -1, -1] + \
       [-1, +1, +1, -1, -1, -1, -1] + \
       [+1, +1, -1, -1, -1, -1, -1] + \
       [+1, +1, +1, +1, +1, +1, +1] + \
       [-1, -1, -1, -1, -1, -1, -1] + \
       [-1, -1, -1, -1, -1, -1, -1]

_three = [+1, +1, +1, +1, +1, +1, +1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [-1, -1, -1, -1, +1, +1, -1] + \
         [-1, -1, -1, +1, +1, -1, -1] + \
         [-1, -1, +1, +1, +1, +1, -1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [-1, +1, +1, +1, +1, +1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1]

_four = [-1, +1, +1, +1, +1, +1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, +1, +1, -1, -1, +1, +1] + \
        [-1, +1, +1, -1, -1, +1, +1] + \
        [+1, +1, +1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, +1, +1, +1] + \
        [-1, +1, +1, +1, +1, +1, +1] + \
        [-1, -1, -1, -1, -1, +1, +1] + \
        [-1, -1, -1, -1, -1, +1, +1]

_five = [-1, +1, +1, +1, +1, +1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [+1, +1, +1, +1, +1, +1, +1] + \
        [+1, +1, -1, -1, -1, -1, -1] + \
        [+1, +1, -1, -1, -1, -1, -1] + \
        [-1, +1, +1, +1, +1, -1, -1] + \
        [-1, -1, +1, +1, +1, +1, -1] + \
        [-1, -1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [-1, +1, +1, +1, +1, +1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1]

_six = [-1, -1, +1, +1, +1, +1, -1] + \
       [-1, +1, +1, -1, -1, -1, -1] + \
       [+1, +1, -1, -1, -1, -1, -1] + \
       [+1, +1, -1, -1, -1, -1, -1] + \
       [+1, +1, +1, +1, +1, +1, -1] + \
       [+1, +1, +1, -1, -1, +1, +1] + \
       [+1, +1, -1, -1, -1, +1, +1] + \
       [+1, +1, -1, -1, -1, +1, +1] + \
       [+1, +1, +1, -1, -1, +1, +1] + \
       [-1, +1, +1, +1, +1, +1, -1] + \
       [-1, -1, -1, -1, -1, -1, -1] + \
       [-1, -1, -1, -1, -1, -1, -1]

_seven = [+1, +1, +1, +1, +1, +1, +1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [-1, -1, -1, -1, +1, +1, -1] + \
         [-1, -1, -1, +1, +1, -1, -1] + \
         [-1, -1, -1, +1, +1, -1, -1] + \
         [-1, -1, +1, +1, -1, -1, -1] + \
         [-1, -1, +1, +1, -1, -1, -1] + \
         [-1, -1, +1, +1, -1, -1, -1] + \
         [-1, -1, +1, +1, -1, -1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1]

_eight = [-1, +1, +1, +1, +1, +1, -1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [-1, +1, +1, +1, +1, +1, -1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [-1, +1, +1, +1, +1, +1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1]

_nine = [-1, +1, +1, +1, +1, +1, -1] + \
        [+1, +1, -1, -1, +1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, +1, +1, +1] + \
        [-1, +1, +1, +1, +1, +1, +1] + \
        [-1, -1, -1, -1, -1, +1, +1] + \
        [-1, -1, -1, -1, -1, +1, +1] + \
        [-1, -1, -1, -1, +1, +1, -1] + \
        [-1, +1, +1, +1, +1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1]


RBF_WEIGHT = np.array([_zero, _one, _two, _three, _four, _five, _six, _seven, _eight, _nine]).transpose()

class RBFLayer(nn.Module):
    def __init__(self, in_features, out_features, init_weight = None):
        super(RBFLayer, self).__init__()
        if init_weight is not None:
            self.register_buffer("weight", torch.tensor(init_weight))
        else:
            self.register_buffer("weight", torch.rand(in_features, out_features))
            
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = (x - self.weight).pow(2).sum(-2) # 计算平方和距离
        return x