#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# 轻量级网络，速度很快，适合于移动端
from torchvision.models import MobileNetV2, Inception3

from torchvision.models import resnet18, resnet34, resnet50


# 分割  不需要预训练模型


class MobileNetV2Define(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = MobileNetV2(num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class MLP(nn.Module):  # ji cheng
    def __init__(self):
        super(MLP, self).__init__()

        # zu zhuang

        # linear = nn.Linear(767*1022*3, 8)
        self.layers = nn.Sequential(
            nn.Linear(32 * 32 * 3, 256),  # 2 ^ n
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )

    def forward(self, x):  # image
        x = x.reshape(-1, 32 * 32 * 3)  # N V  5 32*32*3
        x = self.layers(x)  # out = linear(image)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # 让继承起作用
        self.layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1),
            nn.ReLU(),  # 激活函数  负值变为0 正数保持不变
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 3, 1),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )

        self.linear = nn.Linear(16 * 6 * 6, 8)  # N V

    def forward(self, x):
        x = self.layers(x)
        print("After layers shape:", x.shape)  # 查看经过卷积层后的形状
        x = x.reshape(-1, 16 * 6 * 6)
        print(x.shape)
        x = self.linear(x)
        return x


# 残差网络
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet = resnet18(weights=None)  # weights='DEFAULT'使用默认的预训练权重，如果不想使用，可以改为 weights=None
        self.linear = nn.Linear(1000, 8)

    def forward(self, x):
        x = self.resnet(x) # 通过 ResNet18 提取特征
        x = x.reshape(-1, 1000)
        x = self.linear(x) # 通过全连接层进行分类
        return x


class ResNet50(nn.Module):
    def __init__(self, num_classes=8, pretrained=False):
        super(ResNet50, self).__init__()
        # 如果需要预训练模型，设置 weights='DEFAULT'
        # 如果不需要预训练，设置 weights=None
        self.resnet = resnet50(weights='DEFAULT' if pretrained else None)
        
        # ResNet50的最后一层是1000维输出，我们替换为目标类别数
        self.linear = nn.Linear(1000, num_classes)
        
    def forward(self, x):
        x = self.resnet(x)  # 通过ResNet50提取特征
        x = x.reshape(-1, 1000)
        x = self.linear(x)  # 通过全连接层进行分类
        return x


if __name__ == '__main__':
    # # Tensor  H W C  ->  C H W  -> N V
    # inputs = torch.randn(5, 3, 32, 32)  # image N C H W
    # net = ResNet18()
    # # print(net)
    # out = net(inputs)
    # print(out)
    # print(out.shape)

    # resnet = resnet18(pretrained=True)
    # print(resnet)
    # print(resnet.fc)
    #
    # resnet.fc = nn.Linear(512, 8)
    # print(resnet)


    # 创建模型实例
    net = ResNet50(num_classes=8, pretrained=False)
    
    # 创建测试输入 [batch_size, channels, height, width]
    test_input = torch.randn(4, 3, 224, 224)  # ResNet50标准输入尺寸是224x224
    
    # 前向传播测试
    output = net(test_input)
    
    # 打印输出形状
    print("Output shape:", output.shape)  # 应该输出 [4, 8]


