import torch
from torch import nn


def conv3x3()


class BottleneckBlock(nn.Module):
    def __init__(self,):
        pass

    def forward(self,x):
        return x


class ResidualBasicBlock(nn.Module):
    def __init__(self):
        pass

    def forward(self,x):
        return x


class ResNet18(nn.Module):
    def __init__(self,num_classes=1000):
        self.conv1 = nn.Conv2d(3,64,(7,7),stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # conv layer 2x
        self.conv2x_mp = nn.MaxPool2d((3,3),stride=2,padding=1)


        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,num_classes)
        self.act_final = nn.Softmax()

    def forward(self,x):
        return x


class ResNet34(nn.Module):
    def __init__(self,num_classes=1000):
        self.conv1 = nn.Conv2d(3,64,(7,7),stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # conv layer 2x
        self.conv2x_mp = nn.MaxPool2d((3,3),stride=2,padding=1)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,num_classes)
        self.act_final = nn.Softmax()

    def forward(self,x):
        return x



class ResNet50(nn.Module):
    def __init__(self,num_classes=1000):

        self.conv1 = nn.Conv2d(3,64,(7,7),stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # conv layer 2x
        self.conv2x_mp = nn.MaxPool2d((3,3),stride=2,padding=1)


        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4,num_classes)
        self.act_final = nn.Softmax()

    def forward(self,x):
        return x

class ResNet101(nn.Module):
    def __init__(self,num_classes=1000):
        self.conv1 = nn.Conv2d(3,64,(7,7),stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # conv layer 2x
        self.conv2x_mp = nn.MaxPool2d((3,3),stride=2,padding=1)


        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4,num_classes)
        self.act_final = nn.Softmax()

    def forward(self,x):
        return x

class ResNet152(nn.Module):
    def __init__(self,num_classes=1000):
        self.conv1 = nn.Conv2d(3,64,(7,7),stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # conv layer 2x
        self.conv2x_mp = nn.MaxPool2d((3,3),stride=2,padding=1)


        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4,num_classes)
        self.act_final = nn.Softmax()

    def forward(self,x):
        return x
