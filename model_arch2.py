import torch
from torch import nn

class BottleneckBlock(nn.Module):
    def __init__(self,input_channels,filters,stride=2):
        super().__init__()
        self.stride = stride
        self.input_channels = input_channels
        self.filters = filters

        self.conv1 = nn.Conv2d(input_channels, filters[0], (1,1),stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(filters[0],filters[1],(3,3),padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(filters[1],filters[2],(1,1),bias=False)
        self.bn3 = nn.BatchNorm2d(filters[2])
        self.act3 = nn.ReLU()
        
        if self.stride == 2 or self.input_channels != self.filters[2]:
            self.proj_conv = nn.Conv2d(input_channels, filters[2], (1,1), stride=stride, bias=False)
            self.proj_bn = nn.BatchNorm2d(filters[2])
            self.proj_conv_shortcut = True
            
            nn.init.kaiming_normal_(self.proj_conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.proj_bn.weight,1)
            nn.init.constant_(self.proj_bn.bias,0)
        
        else: 
            self.proj_conv_shortcut = False
        
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        
       
        nn.init.constant_(self.bn1.weight,1)
        nn.init.constant_(self.bn1.bias,0)
        
        nn.init.constant_(self.bn2.weight,1)
        nn.init.constant_(self.bn2.bias,0)
        
        nn.init.constant_(self.bn3.weight,1)
        nn.init.constant_(self.bn3.bias,0)


    def forward(self,x):

        if self.proj_conv_shortcut:
            shortcut = self.proj_conv(x)
            shortcut = self.proj_bn(shortcut)
            #print('shortcut shape=',shortcut.shape)
        else:
            shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        #print('out=',out.shape)
        #print('shortcut shape',shortcut.shape)
            
        out += shortcut

        out = self.act3(out)

        return out


class ResidualBasicBlock(nn.Module):
    def __init__(self,input_channels,filters,stride=2):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(input_channels,filters[0],(3,3),stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(filters[0],filters[1],(3,3),padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.act2 = nn.ReLU()

        if self.stride == 2 or self.input_channels != self.filters[1]:
            self.proj_conv = nn.Conv2d(input_channels, filters[1], (1,1), stride=stride, bias=False)
            self.proj_bn = nn.BatchNorm2d(filters[2])
            self.proj_conv_shortcut = True
            
            nn.init.kaiming_normal_(self.proj_conv.weight, mode='fan_out',nonlinearity='relu')
            nn.init.constant_(self.proj_bn.weight,1)
            nn.init.constant_(self.proj_bn.bias,0)
        

       else: 
            self.proj_conv_shortcut = False

        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out',nonlinearity='relu')
        
        nn.init.constant_(self.bn1.weight,1)
        nn.init.constant_(self.bn1.bias,0)
        
        nn.init.constant_(self.bn2.weight,1)
        nn.init.constant_(self.bn2.bias,0)
        

    def forward(self,x):

        if self.proj_conv_shortcut:
            shortcut = self.proj_conv(x)
            shortcut = self.proj_bn(shortcut)
            #print('shortcut shape=',shortcut.shape)
        else:
            shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += shortcut

        out = self.act2(out)

        return out


class ResNet18(nn.Module):
    def __init__(self,num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,(7,7),stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # conv layer 2x
        self.conv2x_mp = nn.MaxPool2d((3,3),stride=2,padding=1)
        self.conv2x_1 = ResidualBasicBlock(64,[64,64],stride=1)
        self.conv2x_2 = ResidualBasicBlock(64,[64,64],stride=1)

        # conv layer 3x
        self.conv3x_1 = ResidualBasicBlock(64,[128,128],stride=2)
        self.conv3x_2 = ResidualBasicBlock(128,[128,128],stride=1)

        # conv layer 4x
        self.conv4x_1 = ResidualBasicBlock(128,[256,256],stride=2)
        self.conv4x_2 = ResidualBasicBlock(256,[256,256],stride=1)

        # conv layer 5x
        self.conv5x_1 = ResidualBasicBlock(256,[512,512],stride=2)
        self.conv5x_2 = ResidualBasicBlock(512,[512,512],stride=1)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,num_classes)
       
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out',nonlinearity='relu')
        nn.init.constant_(self.bn1.weight,1)
        nn.init.constant_(self.bn1.bias,0)
       

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        #print('conv1x out',out.shape)

        out = self.conv2x_mp(out)
        out = self.conv2x_1(out)
        out = self.conv2x_2(out)
        #print('conv2x out',out.shape)

        out = self.conv3x_1(out)
        out = self.conv3x_2(out)
        #print('conv3x out',out.shape)

        out = self.conv4x_1(out)
        out = self.conv4x_2(out)
        #print('conv4x out',out.shape)

        out = self.conv5x_1(out)
        out = self.conv5x_2(out)
        #print('conv5x out',out.shape)

        out = self.global_avg_pool(out)
        #print('global avg out',out.shape)
        out = torch.flatten(out,1)
        out = self.fc(out)
        #print('final output=',out.shape)
        
        return out




class ResNet34(nn.Module):
    def __init__(self,num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,(7,7),stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # conv layer 2x
        self.conv2x_mp = nn.MaxPool2d((3,3),stride=2,padding=1)
        self.conv2x_1 = ResidualBasicBlock(64,[64,64],stride=1)
        self.conv2x_2 = ResidualBasicBlock(64,[64,64],stride=1)
        self.conv2x_3 = ResidualBasicBlock(64,[64,64],stride=1)


        # conv layer 3x
        self.conv3x_1 = ResidualBasicBlock(64,[128,128],stride=2)
        self.conv3x_2 = ResidualBasicBlock(128,[128,128],stride=1)
        self.conv3x_3 = ResidualBasicBlock(128,[128,128],stride=1)
        self.conv3x_4 = ResidualBasicBlock(128,[128,128],stride=1)

        # conv layer 4x
        self.conv4x_1 = ResidualBasicBlock(128,[256,256],stride=2)
        self.conv4x_2 = ResidualBasicBlock(256,[256,256],stride=1)
        self.conv4x_3 = ResidualBasicBlock(256,[256,256],stride=1)
        self.conv4x_4 = ResidualBasicBlock(256,[256,256],stride=1)
        self.conv4x_5 = ResidualBasicBlock(256,[256,256],stride=1)
        self.conv4x_6 = ResidualBasicBlock(256,[256,256],stride=1)

        # conv layer 5x
        self.conv5x_1 = ResidualBasicBlock(256,[512,512],stride=2)
        self.conv5x_2 = ResidualBasicBlock(512,[512,512],stride=1)
        self.conv5x_3 = ResidualBasicBlock(512,[512,512],stride=1)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,num_classes)
        
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out',nonlinearity='relu')
        nn.init.constant_(self.bn1.weight,1)
        nn.init.constant_(self.bn1.bias,0)
       

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        #print('conv1x out',out.shape)

        out = self.conv2x_mp(out)
        out = self.conv2x_1(out)
        out = self.conv2x_2(out)
        out = self.conv2x_3(out)
        #print('conv2x out',out.shape)

        out = self.conv3x_1(out)
        out = self.conv3x_2(out)
        out = self.conv3x_3(out)
        out = self.conv3x_4(out)
        #print('conv3x out',out.shape)


        out = self.conv4x_1(out)
        out = self.conv4x_2(out)
        out = self.conv4x_3(out)
        out = self.conv4x_4(out)
        out = self.conv4x_5(out)
        out = self.conv4x_6(out)
        #print('conv4x out',out.shape)

        out = self.conv5x_1(out)
        out = self.conv5x_2(out)
        out = self.conv5x_3(out)
        #print('conv5x out',out.shape)

        out = self.global_avg_pool(out)
        #print('global avg pool out',out.shape)
        out = torch.flatten(out,1)
        out = self.fc(out)
        out = self.act_final(out)
        #print('final out',out.shape)
        
        return out


class ResNet50(nn.Module):
    def __init__(self,num_classes=1000):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, (7,7), stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # conv layer 2x
        self.conv2x_mp = nn.MaxPool2d((3,3),stride=2,padding=1)
        self.conv2x_1 = BottleneckBlock(64,[64,64,256],stride=1)
        self.conv2x_2 = BottleneckBlock(256,[64,64,256],stride=1)
        self.conv2x_3 = BottleneckBlock(256,[64,64,256],stride=1)

        # conv layer 3x
        self.conv3x_1 = BottleneckBlock(256,[128,128,512],stride=2)
        self.conv3x_2 = BottleneckBlock(512,[128,128,512],stride=1)
        self.conv3x_3 = BottleneckBlock(512,[128,128,512],stride=1)
        self.conv3x_4 = BottleneckBlock(512,[128,128,512],stride=1)

        # conv layer 4x
        self.conv4x_1 = BottleneckBlock(512,[256,256,1024],stride=2)
        self.conv4x_2 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_3 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_4 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_5 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_6 = BottleneckBlock(1024,[256,256,1024],stride=1)

        # conv layer 5x
        self.conv5x_1 = BottleneckBlock(1024,[512,512,2048],stride=2)
        self.conv5x_2 = BottleneckBlock(2048,[512,512,2048],stride=1)
        self.conv5x_3 = BottleneckBlock(2048,[512,512,2048],stride=1)


        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4,num_classes)
       
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out',nonlinearity='relu')
        nn.init.constant_(self.bn1.weight,1)
        nn.init.constant_(self.bn1.bias,0)
        

   def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        #print('conv1x out',out.shape)
        
        out = self.conv2x_mp(out)
        #print('done layer 2 maxpool')
        out = self.conv2x_1(out)
        #print('conv2x done')
        out = self.conv2x_2(out)
        out = self.conv2x_3(out)
        #print('conv2x out',out.shape)
        
        out = self.conv3x_1(out)
        out = self.conv3x_2(out)
        out = self.conv3x_3(out)
        out = self.conv3x_4(out)
        #print('conv3x out',out.shape)

        out = self.conv4x_1(out)
        out = self.conv4x_2(out)
        out = self.conv4x_3(out)
        out = self.conv4x_4(out)
        out = self.conv4x_5(out)
        out = self.conv4x_6(out)
        #print('conv4x out',out.shape)
        
        
        out = self.conv5x_1(out)
        out = self.conv5x_2(out)
        out = self.conv5x_3(out)
        #print('conv5x out',out.shape)
        
        out = self.global_avg_pool(out)
        #print('global avg pool out',out.shape)
        out = torch.flatten(out,1)
        out = self.fc(out)
        #print('final out',out.shape)

        return out


class ResNet101(nn.Module):
    def __init__(self,num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,(7,7),stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # conv layer 2x
        self.conv2x_mp = nn.MaxPool2d((3,3),stride=2,padding=1)
        self.conv2x_1 = BottleneckBlock(64,[64,64,256],stride=1)
        self.conv2x_2 = BottleneckBlock(256,[64,64,256],stride=1)
        self.conv2x_3 = BottleneckBlock(256,[64,64,256],stride=1)

        # conv layer 3x
        self.conv3x_1 = BottleneckBlock(256, [128,128,512], stride=2)
        self.conv3x_2 = BottleneckBlock(512, [128,128,512], stride=1)
        self.conv3x_3 = BottleneckBlock(512, [128,128,512], stride=1)
        self.conv3x_4 = BottleneckBlock(512, [128,128,512], stride=1)

        # conv layer 4x
        self.conv4x_1 = BottleneckBlock(512,[256,256,1024],stride=2)
        self.conv4x_2 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_3 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_4 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_5 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_6 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_7 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_8 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_9 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_10 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_11 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_12 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_13 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_14 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_15 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_16 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_17 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_18 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_19 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_20 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_21 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_22 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_23 = BottleneckBlock(1024,[256,256,1024],stride=1)

        self.conv5x_1 = BottleneckBlock(1024,[512,512,2048],stride=2)
        self.conv5x_2 = BottleneckBlock(2048,[512,512,2048],stride=1)
        self.conv5x_3 = BottleneckBlock(2048,[512,512,2048],stride=1)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4,num_classes)
       
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out',nonlinearity='relu')
        nn.init.constant_(self.bn1.weight,1)
        nn.init.constant_(self.bn1.bias,0)
        
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        #print('conv1x out',out.shape)

        out = self.conv2x_mp(out)
        out = self.conv2x_1(out)
        out = self.conv2x_2(out)
        out = self.conv2x_3(out)
        #print('conv2x out',out.shape)

        out = self.conv3x_1(out)
        out = self.conv3x_2(out)
        out = self.conv3x_3(out)
        out = self.conv3x_4(out)
        #print('conv3x out',out.shape)


        out = self.conv4x_1(out)
        out = self.conv4x_2(out)
        out = self.conv4x_3(out)
        out = self.conv4x_4(out)
        out = self.conv4x_5(out)
        out = self.conv4x_6(out)
        out = self.conv4x_7(out)
        out = self.conv4x_8(out)
        out = self.conv4x_9(out)
        out = self.conv4x_10(out)
        out = self.conv4x_11(out)
        out = self.conv4x_12(out)
        out = self.conv4x_13(out)
        out = self.conv4x_14(out)
        out = self.conv4x_15(out)
        out = self.conv4x_16(out)
        out = self.conv4x_17(out)
        out = self.conv4x_18(out)
        out = self.conv4x_19(out)
        out = self.conv4x_20(out)
        out = self.conv4x_21(out)
        out = self.conv4x_22(out)
        out = self.conv4x_23(out)
        #print('conv4x out',out.shape)

        out = self.conv5x_1(out)
        out = self.conv5x_2(out)
        out = self.conv5x_3(out)
        #print('conv5x out',out.shape)

        out = self.global_avg_pool(out)
        #print('global pool out',out.shape)
        out = torch.flatten(out,1)
        out = self.fc(out)
        #out = self.act_final(out)
        #print('final out',out.shape)

        return out


class ResNet152(nn.Module):
    def __init__(self,num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,(7,7),stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # conv layer 2x
        self.conv2x_mp = nn.MaxPool2d((3,3),stride=2,padding=1)
        self.conv2x_1 = BottleneckBlock(64,[64,64,256],stride=1)
        self.conv2x_2 = BottleneckBlock(256,[64,64,256],stride=1)
        self.conv2x_3 = BottleneckBlock(256,[64,64,256],stride=1)

        # conv layer 3x
        self.conv3x_1 = BottleneckBlock(256, [128,128,512], stride=2)
        self.conv3x_2 = BottleneckBlock(512, [128,128,512], stride=1)
        self.conv3x_3 = BottleneckBlock(512, [128,128,512], stride=1)
        self.conv3x_4 = BottleneckBlock(512, [128,128,512], stride=1)
        self.conv3x_5 = BottleneckBlock(512, [128,128,512], stride=1)
        self.conv3x_6 = BottleneckBlock(512, [128,128,512], stride=1)
        self.conv3x_7 = BottleneckBlock(512, [128,128,512], stride=1)
        self.conv3x_8 = BottleneckBlock(512, [128,128,512], stride=1)

        # conv layer 4x
        self.conv4x_1 = BottleneckBlock(512,[256,256,1024],stride=2)
        self.conv4x_2 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_3 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_4 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_5 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_6 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_7 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_8 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_9 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_10 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_11 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_12 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_13 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_14 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_15 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_16 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_17 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_18 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_19 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_20 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_21 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_22 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_23 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_24 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_25 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_26 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_27 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_28 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_29 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_30 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_31 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_32 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_33 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_34 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_35 = BottleneckBlock(1024,[256,256,1024],stride=1)
        self.conv4x_36 = BottleneckBlock(1024,[256,256,1024],stride=1)

        # conv5x
        self.conv5x_1 = BottleneckBlock(1024,[512,512,2048],stride=2)
        self.conv5x_2 = BottleneckBlock(2048,[512,512,2048],stride=1)
        self.conv5x_3 = BottleneckBlock(2048,[512,512,2048],stride=1)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4,num_classes)
       
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out',nonlinearity='relu')
        nn.init.constant_(self.bn1.weight,1)
        nn.init.constant_(self.bn1.bias,0)
       

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        #print('conv1x out',out.shape)

        out = self.conv2x_mp(out)
        out = self.conv2x_1(out)
        out = self.conv2x_2(out)
        out = self.conv2x_3(out)
        #print('conv2x out',out.shape)

        out = self.conv3x_1(out)
        out = self.conv3x_2(out)
        out = self.conv3x_3(out)
        out = self.conv3x_4(out)
        #print('conv3x out',out.shape)

        out = self.conv4x_1(out)
        out = self.conv4x_2(out)
        out = self.conv4x_3(out)
        out = self.conv4x_4(out)
        out = self.conv4x_5(out)
        out = self.conv4x_6(out)
        out = self.conv4x_7(out)
        out = self.conv4x_8(out)
        out = self.conv4x_9(out)
        out = self.conv4x_10(out)
        out = self.conv4x_11(out)
        out = self.conv4x_12(out)
        out = self.conv4x_13(out)
        out = self.conv4x_14(out)
        out = self.conv4x_15(out)
        out = self.conv4x_16(out)
        out = self.conv4x_17(out)
        out = self.conv4x_18(out)
        out = self.conv4x_19(out)
        out = self.conv4x_20(out)
        out = self.conv4x_21(out)
        out = self.conv4x_22(out)
        out = self.conv4x_23(out)
        out = self.conv4x_24(out)
        out = self.conv4x_25(out)
        out = self.conv4x_26(out)
        out = self.conv4x_27(out)
        out = self.conv4x_28(out)
        out = self.conv4x_29(out)
        out = self.conv4x_30(out)
        out = self.conv4x_31(out)
        out = self.conv4x_32(out)
        out = self.conv4x_33(out)
        out = self.conv4x_34(out)
        out = self.conv4x_35(out)
        out = self.conv4x_36(out)
        #print('conv4x out',out.shape)

        out = self.conv5x_1(out)
        out = self.conv5x_2(out)
        out = self.conv5x_3(out)
        #print('conv5x out',out.shape)

        out = self.global_avg_pool(out)
        #print('global avg pool out',out.shape)
        out = torch.flatten(out,1)
        out = self.fc(out)
        #print('final out',out.shape)

        return out
