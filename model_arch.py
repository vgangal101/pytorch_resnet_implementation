import torch
from torch import nn

class PlainNetBasicBlock(nn.Module):
    # Output dims is a list
    def __init__(self,in_channels, out_channels,stride=1):

        # set stages 3,4,5 to stride of 2

        super().__init__()

        # layers
        self.conv1 = nn.Conv2d(in_channels,out_channels[0],(3,3),padding=1,stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.act1_relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels[0],out_channels[1],(3,3),padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.act2_relu = nn.ReLU()

    def forward(self,input_tensor):

        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.act1_relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2_relu(out)

        return out


class plain_ResNet18(nn.Module):
    def __init__(self):

        super().__init__()
        # conv layer 1
        self.conv1 = nn.Conv2d(3,64,(7,7),stride=2,padding=3)

        # conv layer 2x
        self.conv2x_mp = nn.MaxPool2d((3,3),stride=2,padding=1)
        self.conv2x_block1 = PlainNetBasicBlock(64,[64,64])
        self.conv2x_block2 = PlainNetBasicBlock(64,[64,64])

        # conv layer 3x
        self.conv3x_1 = PlainNetBasicBlock(64,[128,128],stride=2)
        self.conv3x_2 = PlainNetBasicBlock(128,[128,128],)

        # conv layer 4x
        self.conv4x_1 = PlainNetBasicBlock(128,[256,256],stride=2)
        self.conv4x_2 = PlainNetBasicBlock(256,[256,256])

        # conv layer 5x
        self.conv5x_1 = PlainNetBasicBlock(256,[512,512],stride=2)
        self.conv5x_2 = PlainNetBasicBlock(512,[512,512])

        # last layers
        self.global_avg_pool = nn.AvgPool2d((7,7))
        self.fc = nn.Linear(512,1000)

    def forward(self,x):

        out = self.conv1(x)

        out = self.conv2x_mp(out)
        out = self.conv2x_block1(out)
        out = self.conv2x_block2(out)

        out = self.conv3x_1(out)
        out = self.conv3x_2(out)


        out = self.conv4x_1(out)
        out = self.conv4x_2(out)


        out = self.conv5x_1(out)
        out = self.conv5x_2(out)


        out = self.global_avg_pool(out)
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)


        return out

class plain_ResNet34(nn.Module):
    def __init__(self):

        super().__init__()
        # conv layer 1
        self.conv1 = nn.Conv2d(3,64,(7,7),stride=2,padding=3)

        # conv layer 2x
        self.conv2x_mp = nn.MaxPool2d((3,3),stride=2,padding=1)
        self.conv2x_block1 = PlainNetBasicBlock(64,[64,64])
        self.conv2x_block2 = PlainNetBasicBlock(64,[64,64])
        self.conv2x_block3 = PlainNetBasicBlock(64,[64,64])


        # conv layer 3x
        self.conv3x_1 = PlainNetBasicBlock(64,[128,128],stride=2)
        self.conv3x_2 = PlainNetBasicBlock(128,[128,128])
        self.conv3x_3 = PlainNetBasicBlock(128,[128,128])
        self.conv3x_4 = PlainNetBasicBlock(128,[128,128])

        # conv layer 4x
        self.conv4x_1 = PlainNetBasicBlock(128,[256,256],stride=2)
        self.conv4x_2 = PlainNetBasicBlock(256,[256,256])
        self.conv4x_3 = PlainNetBasicBlock(256,[256,256])
        self.conv4x_4 = PlainNetBasicBlock(256,[256,256])
        self.conv4x_5 = PlainNetBasicBlock(256,[256,256])
        self.conv4x_6 = PlainNetBasicBlock(256,[256,256])


        # conv layer 5x
        self.conv5x_1 = PlainNetBasicBlock(256,[512,512],stride=2)
        self.conv5x_2 = PlainNetBasicBlock(512,[512,512])
        self.conv5x_3 = PlainNetBasicBlock(512,[512,512])

        # last layers
        self.global_avg_pool = nn.AvgPool2d((7,7))
        self.fc = nn.Linear(512,1000)
        self.act_final = nn.Softmax()

    def forward(self,x):

        out = self.conv1(x)
        #print('conv1x out=',out.shape)

        out = self.conv2x_mp(out)
        out = self.conv2x_block1(out)
        out = self.conv2x_block2(out)
        #print('conv2x out=',out.shape)

        out = self.conv3x_1(out)
        out = self.conv3x_2(out)
        out = self.conv3x_3(out)
        out = self.conv3x_4(out)
        #print('conv3x out=',out.shape)


        out = self.conv4x_1(out)
        out = self.conv4x_2(out)
        out = self.conv4x_3(out)
        out = self.conv4x_4(out)
        out = self.conv4x_5(out)
        out = self.conv4x_6(out)
        #print('conv4x out=',out.shape)


        out = self.conv5x_1(out)
        out = self.conv5x_2(out)
        out = self.conv5x_3(out)

        #print('conv5x out=',out.shape)


        out = self.global_avg_pool(out)
        #print('after global avg pool=',out.shape)
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)
        #print('after fc=',out.shape)

        out = self.act_final(out)

        return out

class BuildingBlockResidual_Identity(nn.Module):
    # Output dims is a list
    def __init__(self,in_channels, out_channels):

        # set stages 3,4,5 to stride of 2

        super().__init__()

        # layers
        self.conv1 = nn.Conv2d(in_channels,out_channels[0],(3,3),padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.act1_relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels[0],out_channels[1],(3,3),padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.act2_relu = nn.ReLU()

        torch.nn.init.kaiming_normal_(self.conv1.weight)
        #torch.nn.init.kaiming_normal_(self.conv1.bias)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        #torch.nn.init.kaiming_normal_(self.conv2.bias)



    def forward(self,input_tensor):
        short_cut = input_tensor

        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.act1_relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2_relu(out)

        out += short_cut

        return out

class BuildingBlockResidual_ConvBlock(nn.Module):

  def __init__(self,in_channels, out_channels,stride=1):
        super().__init__()
        self.proj_conv = nn.Conv2d(in_channels,out_channels[1],(1,1),stride=stride)
        self.proj_bn = nn.BatchNorm2d(out_channels[1])

        self.conv1 = nn.Conv2d(in_channels,out_channels[0],(3,3),padding=1,stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.act1_relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels[0],out_channels[1],(3,3),padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.act2_relu = nn.ReLU()


        torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        #torch.nn.init.kaiming_normal_(self.conv1.bias, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        #torch.nn.init.kaiming_normal_(self.conv2.bias, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.proj_conv.weight, mode='fan_out', nonlinearity='relu')
        #torch.nn.init.kaiming_normal_(self.proj_conv.bias, nonlinearity='relu')


  def forward(self,input_tensor):

      out = self.conv1(input_tensor)
      out = self.bn1(out)
      out = self.act1_relu(out)

      out = self.conv2(out)
      out = self.bn2(out)
      shortcut = self.proj_conv(input_tensor)
      shortcut = self.proj_bn(shortcut)

      out += shortcut
      out = self.act2_relu(out)

      return out


class ResNet18(nn.Module):
  def __init__(self,num_classes=1000):
    super().__init__()
    # conv layer 1
    # does bias need to be false ??
    self.conv1 = nn.Conv2d(3,64,(7,7),stride=2,padding=3)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu1 = nn.ReLU()


    # conv layer 2x
    self.conv2x_mp = nn.MaxPool2d((3,3),stride=2,padding=1)
    self.conv2x_block1 = BuildingBlockResidual_ConvBlock(64,[64,64],stride=1)
    self.conv2x_block2 = BuildingBlockResidual_Identity(64,[64,64])

    # conv layer 3x
    self.conv3x_1 = BuildingBlockResidual_ConvBlock(64,[128,128],stride=2)
    self.conv3x_2 = BuildingBlockResidual_Identity(128,[128,128])

    # conv layer 4x
    self.conv4x_1 = BuildingBlockResidual_ConvBlock(128,[256,256],stride=2)
    self.conv4x_2 = BuildingBlockResidual_Identity(256,[256,256])

    # conv layer 5x
    self.conv5x_1 = BuildingBlockResidual_ConvBlock(256,[512,512],stride=2)
    self.conv5x_2 = BuildingBlockResidual_Identity(512,[512,512])

    # last layers
    self.global_avg_pool = nn.AvgPool2d((7,7))
    self.fc = nn.Linear(512,num_classes)
    self.act_final = nn.Softmax()

    torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
    #torch.nn.init.kaiming_normal_(self.conv1.bias, nonlinearity='relu')


  def forward(self,x):

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)
    #print('conv1x out=',out.shape)

    out = self.conv2x_mp(out)
    out = self.conv2x_block1(out)
    out = self.conv2x_block2(out)
    #print('conv2x out=',out.shape)

    out = self.conv3x_1(out)
    out = self.conv3x_2(out)
    #print('conv3x out=',out.shape)


    out = self.conv4x_1(out)
    out = self.conv4x_2(out)
    #print('conv4x out=',out.shape)


    out = self.conv5x_1(out)
    out = self.conv5x_2(out)
    #print('conv5x out=',out.shape)


    out = self.global_avg_pool(out)
    #print('after global avg pool=',out.shape)
    out = out.reshape(out.shape[0],-1)
    out = self.fc(out)
    #print('after fc=',out.shape)

    out = self.act_final(out)

    return out

class ResNet34(nn.Module):
  def __init__(self, num_classes=1000):
    super().__init__()
    # conv layer 1
    self.conv1 = nn.Conv2d(3,64,(7,7),stride=2,padding=3)

    # conv layer 2x
    self.conv2x_mp = nn.MaxPool2d((3,3),stride=2,padding=1)
    self.conv2x_block1 = BuildingBlockResidual_ConvBlock(64,[64,64],stride=1)
    self.conv2x_block2 = BuildingBlockResidual_Identity(64,[64,64])
    self.conv2x_block3 = BuildingBlockResidual_Identity(64,[64,64])

    # conv layer 3x
    self.conv3x_1 = BuildingBlockResidual_ConvBlock(64,[128,128],stride=2)
    self.conv3x_2 = BuildingBlockResidual_Identity(128,[128,128])
    self.conv3x_3 = BuildingBlockResidual_Identity(128,[128,128])
    self.conv3x_4 = BuildingBlockResidual_Identity(128,[128,128])

    # conv layer 4x
    self.conv4x_1 = BuildingBlockResidual_ConvBlock(128,[256,256],stride=2)
    self.conv4x_2 = BuildingBlockResidual_Identity(256,[256,256])
    self.conv4x_3 = BuildingBlockResidual_Identity(256,[256,256])
    self.conv4x_4 = BuildingBlockResidual_Identity(256,[256,256])
    self.conv4x_5 = BuildingBlockResidual_Identity(256,[256,256])
    self.conv4x_6 = BuildingBlockResidual_Identity(256,[256,256])

    # conv layer 5x
    self.conv5x_1 = BuildingBlockResidual_ConvBlock(256,[512,512],stride=2)
    self.conv5x_2 = BuildingBlockResidual_Identity(512,[512,512])
    self.conv5x_3 = BuildingBlockResidual_Identity(512,[512,512])


    # last layers
    self.global_avg_pool = nn.AvgPool2d((7,7))
    self.fc = nn.Linear(512,num_classes)
    self.act_final = nn.Softmax()

    torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
    #torch.nn.init.kaiming_normal_(self.conv1.bias, nonlinearity='relu')


  def forward(self,x):

        out = self.conv1(x)
        #print('conv1x out=',out.shape)

        out = self.conv2x_mp(out)
        out = self.conv2x_block1(out)
        out = self.conv2x_block2(out)
        out = self.conv2x_block3(out)
        #print('conv2x out=',out.shape)

        out = self.conv3x_1(out)
        out = self.conv3x_2(out)
        out = self.conv3x_2(out)
        out = self.conv3x_4(out)
        #print('conv3x out=',out.shape)


        out = self.conv4x_1(out)
        out = self.conv4x_2(out)
        out = self.conv4x_3(out)
        out = self.conv4x_4(out)
        out = self.conv4x_5(out)
        out = self.conv4x_6(out)
        #print('conv4x out=',out.shape)


        out = self.conv5x_1(out)
        out = self.conv5x_2(out)
        out = self.conv5x_3(out)
        #print('conv5x out=',out.shape)


        out = self.global_avg_pool(out)
        #print('after global avg pool=',out.shape)
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)
        #print('after fc=',out.shape)

        out = self.act_final(out)

        return out


# bottle neck block

class BottleNeckBlock_Identity(nn.Module):
  def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels[0],(1,1),padding='valid')
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels[0],out_channels[1],(3,3),padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels[1],out_channels[2],(1,1),padding='valid')
        self.bn3 = nn.BatchNorm2d(out_channels[2])
        self.act3 = nn.ReLU()

        torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        #torch.nn.init.kaiming_normal_(self.conv1.bias, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        #torch.nn.init.kaiming_normal_(self.conv2.bias, mode='fan_out', nonlinearity='relu')
        #torch.nn.init.kaiming_normal_(self.conv3.bias, mode='fan_out', nonlinearity='relu')

        #torch.nn.init.kaiming_normal_(self.proj_conv.weight, mode='fan_out', nonlinearity='relu')
        #torch.nn.init.kaiming_normal_(self.proj_conv.bias, nonlinearity='relu')

  def forward(self,input_tensor):

        #print('input_tensor dims=', input_tensor.shape)
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += input_tensor
        out = self.act3(out)

        return out

class BottleNeckBlock_Conv(nn.Module):
  def __init__(self,in_channels,out_channels,stride=2):

        """
          self.proj_conv = nn.Conv2d(in_channels,out_channels[1],(1,1),stride=stride)
          self.proj_bn = nn.BatchNorm2d(out_channels[1])
          self.conv1 = nn.Conv2d(in_channels,out_channels[0],(3,3),padding=1,stride=stride)
        """
        super().__init__()
        self.proj_conv = nn.Conv2d(in_channels,out_channels[2],(1,1),stride=stride)
        self.proj_bn = nn.BatchNorm2d(out_channels[2])

        # main conv layers
        self.conv1 = nn.Conv2d(in_channels, out_channels[0], (1,1), stride=stride, padding='valid')
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels[0],out_channels[1], (3,3), padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels[1],out_channels[2],(1,1),padding='valid')
        self.bn3 = nn.BatchNorm2d(out_channels[2])
        self.act3 = nn.ReLU()

        torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        #print('applied conv1 weight init')
        #torch.nn.init.kaiming_normal_(self.conv1.bias, mode='fan_out',nonlinearity='relu')
        #print('applied conv1 bias  init')
        torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        #print('applied conv2 weight  init')
        torch.nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        #print('applied conv3 weight  init')
        #torch.nn.init.kaiming_normal_(self.conv2.bias, mode='fan_out', nonlinearity='relu')
        #print('applied conv2 bias  init')
        torch.nn.init.kaiming_normal_(self.proj_conv.weight, mode='fan_out', nonlinearity='relu')
        #print('applied proj conv weight  init')
        #torch.nn.init.kaiming_normal_(self.proj_conv.bias,mode='fan_out',nonlinearity='relu')
        #print('applied proj_conv bias  init')

  def forward(self,input_tensor):
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)


        out = self.conv3(out)
        out = self.bn3(out)


        shortcut = self.proj_conv(input_tensor)
        shortcut = self.proj_bn(shortcut)

        out += shortcut
        out = self.act3(out)


        return out


class ResNet50(nn.Module):
  def __init__(self,num_classes=1000):
        super().__init__()
        # conv layer 1
        self.conv1 = nn.Conv2d(3,64,(7,7),stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()

        # conv layer 2x
        self.conv2x_mp = nn.MaxPool2d((3,3),stride=2,padding=1)
        self.conv2x_block1 = BottleNeckBlock_Conv(64,[64,64,256],stride=1)
        self.conv2x_block2 = BottleNeckBlock_Identity(256,[64,64,256])
        self.conv2x_block3 = BottleNeckBlock_Identity(256,[64,64,256])

        # conv layer 3x
        self.conv3x_1 = BottleNeckBlock_Conv(256,[128,128,512],stride=2)
        self.conv3x_2 = BottleNeckBlock_Identity(512,[128,128,512])
        self.conv3x_3 = BottleNeckBlock_Identity(512,[128,128,512])
        self.conv3x_4 = BottleNeckBlock_Identity(512,[128,128,512])

        # conv layer 4x
        self.conv4x_1 = BottleNeckBlock_Conv(512,[256,256,1024],stride=2)
        self.conv4x_2 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_3 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_4 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_5 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_6 = BottleNeckBlock_Identity(1024,[256,256,1024])

        # conv layer 5x
        self.conv5x_1 = BottleNeckBlock_Conv(1024,[512,512,2048],stride=2)
        self.conv5x_2 = BottleNeckBlock_Identity(2048,[512,512,2048])
        self.conv5x_3 = BottleNeckBlock_Identity(2048,[512,512,2048])

        # last layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048,num_classes)
        self.act_final = nn.Softmax()

        torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        #torch.nn.init.kaiming_normal_(self.conv1.bias, mode='fan_out', nonlinearity='relu')


  def forward(self,x):

        #print('input type=',type(x))
        #print('input=',x)
        #print('input_shape=',x.shape)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        #print('conv1x out=',out.shape)

        out = self.conv2x_mp(out)
        out = self.conv2x_block1(out)
        out = self.conv2x_block2(out)

        out = self.conv2x_block3(out)
        #print('conv2x out=',out.shape)


        out = self.conv3x_1(out)
        out = self.conv3x_2(out)
        out = self.conv3x_2(out)
        out = self.conv3x_4(out)
        #print('conv3x out=',out.shape)


        out = self.conv4x_1(out)
        out = self.conv4x_2(out)
        out = self.conv4x_3(out)
        out = self.conv4x_4(out)
        out = self.conv4x_5(out)
        out = self.conv4x_6(out)
        #print('conv4x out=',out.shape)


        out = self.conv5x_1(out)
        out = self.conv5x_2(out)
        out = self.conv5x_3(out)
        #print('conv5x out=',out.shape)


        out = self.global_avg_pool(out)
        #print('global out.shape=',out.shape)
        #print('after global avg pool=',out.shape)
        
        out = torch.flatten(out,1)
        
        out = self.fc(out)
        #print('after fc=',out.shape)

        out = self.act_final(out)

        return out

class ResNet101(nn.Module):
  def __init__(self,num_classes=1000):
        super().__init__()
        # conv layer 1
        self.conv1 = nn.Conv2d(3,64,(7,7),stride=2,padding=3)

        # conv layer 2x
        self.conv2x_mp = nn.MaxPool2d((3,3),stride=2,padding=1)
        self.conv2x_block1 = BottleNeckBlock_Conv(64,[64,64,256],stride=1)
        self.conv2x_block2 = BottleNeckBlock_Identity(256,[64,64,256])
        self.conv2x_block3 = BottleNeckBlock_Identity(256,[64,64,256])

        # conv layer 3x (need 4 blocks here)
        self.conv3x_1 = BottleNeckBlock_Conv(256,[128,128,512],stride=2)
        self.conv3x_2 = BottleNeckBlock_Identity(512,[128,128,512])
        self.conv3x_3 = BottleNeckBlock_Identity(512,[128,128,512])
        self.conv3x_4 = BottleNeckBlock_Identity(512,[128,128,512])

        # conv layer 4x ( need  23 blocks )
        self.conv4x_1 = BottleNeckBlock_Conv(512,[256,256,1024],stride=2)
        self.conv4x_2 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_3 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_4 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_5 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_6 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_7 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_8 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_9 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_10 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_11 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_12 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_13 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_14 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_15 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_16 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_17 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_18 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_19 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_20 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_21 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_22 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_23 = BottleNeckBlock_Identity(1024,[256,256,1024])

        # conv layer 5x (3  blocks )
        self.conv5x_1 = BottleNeckBlock_Conv(1024,[512,512,2048],stride=2)
        self.conv5x_2 = BottleNeckBlock_Identity(2048,[512,512,2048])
        self.conv5x_3 = BottleNeckBlock_Identity(2048,[512,512,2048])

        # last layers
        self.global_avg_pool = nn.AvgPool2d((7,7))
        self.fc = nn.Linear(2048,num_classes)
        self.act_final = nn.Softmax()

        torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        #torch.nn.init.kaiming_normal_(self.conv1.bias, mode='fan_out', nonlinearity='relu')


  def forward(self,x):
        out = self.conv1(x)
        print('conv1x out=',out.shape)

        out = self.conv2x_mp(out)
        out = self.conv2x_block1(out)
        out = self.conv2x_block2(out)

        out = self.conv2x_block3(out)
        print('conv2x out=',out.shape)


        out = self.conv3x_1(out)
        out = self.conv3x_2(out)
        out = self.conv3x_2(out)
        out = self.conv3x_4(out)
        print('conv3x out=',out.shape)


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
        print('conv4x out=',out.shape)


        out = self.conv5x_1(out)
        out = self.conv5x_2(out)
        out = self.conv5x_3(out)
        print('conv5x out=',out.shape)


        out = self.global_avg_pool(out)
        print('after global avg pool=',out.shape)
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)
        print('after fc=',out.shape)

        out = self.act_final(out)

        return out



class ResNet152(nn.Module):
  def __init__(self,num_classes=1000):
        super().__init__()
        # conv layer 1
        self.conv1 = nn.Conv2d(3,64,(7,7),stride=2,padding=3)

        # conv layer 2x
        self.conv2x_mp = nn.MaxPool2d((3,3),stride=2,padding=1)
        self.conv2x_block1 = BottleNeckBlock_Conv(64,[64,64,256],stride=1)
        self.conv2x_block2 = BottleNeckBlock_Identity(256,[64,64,256])
        self.conv2x_block3 = BottleNeckBlock_Identity(256,[64,64,256])

        # conv layer 3x (need 4 blocks here)
        self.conv3x_1 = BottleNeckBlock_Conv(256,[128,128,512],stride=2)
        self.conv3x_2 = BottleNeckBlock_Identity(512,[128,128,512])
        self.conv3x_3 = BottleNeckBlock_Identity(512,[128,128,512])
        self.conv3x_4 = BottleNeckBlock_Identity(512,[128,128,512])

        # conv layer 4x ( need  23 blocks )
        self.conv4x_1 = BottleNeckBlock_Conv(512,[256,256,1024],stride=2)
        self.conv4x_2 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_3 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_4 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_5 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_6 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_7 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_8 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_9 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_10 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_11 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_12 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_13 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_14 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_15 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_16 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_17 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_18 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_19 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_20 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_21 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_22 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_23 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_24 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_25 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_26 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_27 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_28 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_29 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_30 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_31 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_32 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_33 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_34 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_35 = BottleNeckBlock_Identity(1024,[256,256,1024])
        self.conv4x_36 = BottleNeckBlock_Identity(1024,[256,256,1024])

        # conv layer 5x (3  blocks )
        self.conv5x_1 = BottleNeckBlock_Conv(1024,[512,512,2048],stride=2)
        self.conv5x_2 = BottleNeckBlock_Identity(2048,[512,512,2048])
        self.conv5x_3 = BottleNeckBlock_Identity(2048,[512,512,2048])

        # last layers
        self.global_avg_pool = nn.AvgPool2d((7,7))
        self.fc = nn.Linear(2048,num_classes)
        self.act_final = nn.Softmax()

        torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        #torch.nn.init.kaiming_normal_(self.conv1.bias, mode='fan_out', nonlinearity='relu')


  def forward(self,x):
        out = self.conv1(x)
        print('conv1x out=',out.shape)

        out = self.conv2x_mp(out)
        out = self.conv2x_block1(out)
        out = self.conv2x_block2(out)

        out = self.conv2x_block3(out)
        print('conv2x out=',out.shape)


        out = self.conv3x_1(out)
        out = self.conv3x_2(out)
        out = self.conv3x_2(out)
        out = self.conv3x_4(out)
        print('conv3x out=',out.shape)


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
        print('conv4x out=',out.shape)


        out = self.conv5x_1(out)
        out = self.conv5x_2(out)
        out = self.conv5x_3(out)
        print('conv5x out=',out.shape)


        out = self.global_avg_pool(out)
        print('after global avg pool=',out.shape)
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)
        print('after fc=',out.shape)

        out = self.act_final(out)

        return out
