# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 22:38:42 2022

@author: lenovo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms as T

import matplotlib.pyplot as plt
import numpy as np

from LeNet5 import LeNet5
        

lossList = [] # 损失
trainError = [] # 训练误差
testError = [] # 测试误差

'''
损失函数
'''
def loss_fn(pred, label):
    if(label.dim()==1):
        return pred[torch.arange(pred.size(0)), label]
    else:
        return pred[torch.arange(pred.size(0)), label.squeeze()]

'''
训练函数
'''
def train(epochs, model, optimizer, scheduler: bool, loss_fn, trainSet, testSet):

    trainNum = len(trainSet) # 训练集样本数
    testNum = len(testSet) # 测试集样本数
    
    
    for epoch in range(epochs):
        lossSum = 0.0 # 初始化误差和
        print("epoch: {:02d} / {:d}".format(epoch+1, epochs))
        
        for idx, (img, label) in enumerate(trainSet):
            x = img.unsqueeze(0).to(device)
            y = torch.tensor([label], dtype = torch.long).to(device)
            
            out = model(x)
            optimizer.zero_grad()
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            
            lossSum += loss.item()
            if (idx + 1) % 2000 == 0: print("sample: {:05d} / {:d} --> loss: {:.4f}".format(idx+1, trainNum, loss.item()))
        
        lossList.append(lossSum / trainNum)
        
        with torch.no_grad():
            errorNum = 0
            for img, label in trainSet:
                x = img.unsqueeze(0).to(device)
                out = model(x)
                _, pred_y = out.min(dim = 1)
                if(pred_y != label): errorNum += 1
            trainError.append(errorNum / trainNum)
            
            errorNum = 0
            for img, label in testSet:
                x = img.unsqueeze(0).to(device)
                out = model(x)
                _, pred_y = out.min(dim = 1)
                if(pred_y != label): errorNum += 1
            testError.append(errorNum / testNum)
        
        if scheduler == True:
            if epoch < 5:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1.0e-3
            elif epoch < 10:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 5.0e-4
            elif epoch < 15:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 2.0e-4
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1.0e-4

    torch.save(model.state_dict(), 'E:/xio习/计算机视觉/LeNet-5/epoch-{:d}_loss-{:.6f}_error-{:.2%}.pth'.format(epochs, lossList[-1], testError[-1]))

        
'''
图像预处理：使图像转化为像素值范围[-0.1, 1.175]，方差为1的Tensor
'''
picProcessor = T.Compose([
    T.ToTensor(), # 将灰度范围从0-255变换到0-1之间
    T.Normalize(
        mean = [0.1/1.275],
        std = [1.0/1.275])]) # 使最小值变成(0-0.1)/1=-0.1，(0-1.275)/1.275=-1，最大值变成(1-0.1)/1=0.9

'''
加载数据集
dataPath：数据集地址
mnistTrain：训练集数据
mnistTest:测试集数据
'''
dataPath = r'E:\xio习\计算机视觉\LeNet-5'
mnistTrain = datasets.MNIST(dataPath, train=True, download=False, transform=picProcessor) #需要下载将download改为True
mnistTest = datasets.MNIST(dataPath, train=False, download=False, transform=picProcessor)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') # 若有GPU，使用GPU进行加速

if __name__ == '__main__':
    model = LeNet5().to(device) # model：LeNet-5对象
    optimizer = optim.SGD(model.parameters(), lr = 1.0e-3)
    
    scheduler = True
    
    epochs = 25
    
    train(epochs, model, optimizer, scheduler, loss_fn, mnistTrain, mnistTest)
    plt.subplot(1, 3, 1)
    plt.plot(lossList)
    plt.subplot(1, 3, 2)
    plt.plot(trainError)
    plt.plot(testError)
    plt.show()
    