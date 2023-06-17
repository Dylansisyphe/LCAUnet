"""
Author: Zhuo Su, Wenzhe Liu
Date: Feb 18, 2021
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .edge_ops import Conv2d
from .edge_config import config_model, config_model_converted

class CSAM(nn.Module):
    """
    Compact Spatial Attention Module
    消除背景噪音的影响
    """
    def __init__(self, channels):
        super(CSAM, self).__init__()

        mid_channels = 4 #降维channel，也是通过csam模块后的通道数，该数小于输入channel数，以此降低运算量
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        y = self.relu1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sigmoid(y)

        return x * y

class CDCM(nn.Module):
    """
    Compact Dilation Convolution based Module
    联合使用不同膨胀率的空洞卷积，来增强多尺度的边缘信息
    """
    def __init__(self, in_channels, out_channels):
        super(CDCM, self).__init__()

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False) #注意空洞卷积对应膨胀率的填充
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False)
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False)
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False)
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        return x1 + x2 + x3 + x4


class MapReduce(nn.Module):
    """
    Reduce feature maps into a single edge map
    最后通过一个简单的卷积操作生成边缘图像的类
    """
    def __init__(self, channels):
        super(MapReduce, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class PDCBlock(nn.Module):
    """
    Block_x_y块类
    conv1:depth-wise的PDC卷积
    shortcut:跳跃连接，只有在特征图尺寸缩小的时候使用跳跃连接
    """
    def __init__(self, pdc, inplane, ouplane, stride=1):
        super(PDCBlock, self).__init__()
        self.stride=stride
        if self.stride > 1:
            #将池化操作嵌入到通用block中，根据步长选择性使用
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        self.conv1 = Conv2d(pdc, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y

class PDCBlock_converted(nn.Module):
    """
    CPDC, APDC can be converted to vanilla 3x3 convolution
    RPDC can be converted to vanilla 5x5 convolution
    """
    def __init__(self, pdc, inplane, ouplane, stride=1):
        super(PDCBlock_converted, self).__init__()
        self.stride=stride

        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        if pdc == 'rd':
            self.conv1 = nn.Conv2d(inplane, inplane, kernel_size=5, padding=2, groups=inplane, bias=False)
        else:
            self.conv1 = nn.Conv2d(inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y

class PiDiNet(nn.Module):
    """
    sa:是否在PiDiNet中使用CSAM
    dil:是否在PiDiNet中使用CDCM
    inplane:初始的输出通道个数，该参数对应通道数的变化
    stride:对应特征图尺寸的变化，如果使用，在每一阶段block的初始block使用
    pdcs:网络中所有像素差卷积的参数数组
    convert:boolean,是否使用卷积来代替普通的像素差操作
    """
    def __init__(self, inplane, pdcs, dil=None, sa=False, convert=False):
        super(PiDiNet, self).__init__()
        self.sa = sa
        if dil is not None:
            assert isinstance(dil, int), 'dil should be an int'
        self.dil = dil

        self.fuseplanes = [] #记录各block最后输出的通道数，方便进行后续的CDCM和CSAM操作，并进行融合

        self.inplane = inplane
        if convert:
            if pdcs[0] == 'rd':
                init_kernel_size = 5
                init_padding = 2
            else:
                init_kernel_size = 3
                init_padding = 1
            self.init_block = nn.Conv2d(3, self.inplane,
                    kernel_size=init_kernel_size, padding=init_padding, bias=False)
            block_class = PDCBlock_converted
        else:
            #之所以会有init_block是因为其输入的通道数来自于数据，而不是推导，这里默认输入数据为含RGB三个通道的图像
            self.init_block = Conv2d(pdcs[0], 3, self.inplane, kernel_size=3, padding=1)
            block_class = PDCBlock

        self.block1_1 = block_class(pdcs[1], self.inplane, self.inplane)
        self.block1_2 = block_class(pdcs[2], self.inplane, self.inplane)
        self.block1_3 = block_class(pdcs[3], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane) # C

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block2_1 = block_class(pdcs[4], inplane, self.inplane, stride=2)
        self.block2_2 = block_class(pdcs[5], self.inplane, self.inplane)
        self.block2_3 = block_class(pdcs[6], self.inplane, self.inplane)
        self.block2_4 = block_class(pdcs[7], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane) # 2C

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block3_1 = block_class(pdcs[8], inplane, self.inplane, stride=2)
        self.block3_2 = block_class(pdcs[9], self.inplane, self.inplane)
        self.block3_3 = block_class(pdcs[10], self.inplane, self.inplane)
        self.block3_4 = block_class(pdcs[11], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane) # 4C

        self.block4_1 = block_class(pdcs[12], self.inplane, self.inplane, stride=2)
        self.block4_2 = block_class(pdcs[13], self.inplane, self.inplane)
        self.block4_3 = block_class(pdcs[14], self.inplane, self.inplane)
        self.block4_4 = block_class(pdcs[15], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane) # 4C

        self.conv_reduces = nn.ModuleList()
        if self.sa and self.dil is not None:
            self.attentions = nn.ModuleList()
            self.dilations = nn.ModuleList()
            for i in range(4):
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                self.attentions.append(CSAM(self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        elif self.sa: #只使用CSAM模块
            self.attentions = nn.ModuleList()
            for i in range(4):
                self.attentions.append(CSAM(self.fuseplanes[i]))
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))
        elif self.dil is not None: #只使用CDCM模块
            self.dilations = nn.ModuleList()
            for i in range(4):
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        else:
            for i in range(4):
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))

        self.classifier = nn.Conv2d(4, 1, kernel_size=1) # has bias，四个不同尺度的特征图对应四个通道，最后只输出一个特征图，对应最后一个通道
        nn.init.constant_(self.classifier.weight, 0.25)
        nn.init.constant_(self.classifier.bias, 0)

        print('initialization done')

    def get_weights(self):
        conv_weights = []
        bn_weights = []
        relu_weights = []
        for pname, p in self.named_parameters():
            if 'bn' in pname:
                bn_weights.append(p)
            elif 'relu' in pname:
                relu_weights.append(p)
            else:
                conv_weights.append(p)

        return conv_weights, bn_weights, relu_weights

    def forward(self, x):
        #pytorch中Tensor维度的顺序为BCHW
        #pytorch是在哪里对数据维度的顺序做转换的？
        H, W = x.size()[2:]

        x = self.init_block(x)

        x1 = self.block1_1(x)
        x1 = self.block1_2(x1)
        x1 = self.block1_3(x1)

        x2 = self.block2_1(x1)
        x2 = self.block2_2(x2)
        x2 = self.block2_3(x2)
        x2 = self.block2_4(x2)

        x3 = self.block3_1(x2)
        x3 = self.block3_2(x3)
        x3 = self.block3_3(x3)
        x3 = self.block3_4(x3)

        x4 = self.block4_1(x3)
        x4 = self.block4_2(x4)
        x4 = self.block4_3(x4)
        x4 = self.block4_4(x4)

        x_stages_features = [x1, x2, x3, x4]
        x_fuses = []
        if self.sa and self.dil is not None:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.attentions[i](self.dilations[i](xi)))
        elif self.sa:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.attentions[i](xi))
        elif self.dil is not None:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.dilations[i](xi))
        else:
            x_fuses = [x1, x2, x3, x4]

        outputs = []
        e1 = self.conv_reduces[0](x_fuses[0])
        e1 = F.interpolate(e1, (H, W), mode="bilinear", align_corners=False)
        #e1的shape BCHW,C为1?

        e2 = self.conv_reduces[1](x_fuses[1])
        e2 = F.interpolate(e2, (H, W), mode="bilinear", align_corners=False)

        e3 = self.conv_reduces[2](x_fuses[2])
        e3 = F.interpolate(e3, (H, W), mode="bilinear", align_corners=False)

        e4 = self.conv_reduces[3](x_fuses[3])
        e4 = F.interpolate(e4, (H, W), mode="bilinear", align_corners=False)

        outputs = [e1, e2, e3, e4]

        output = self.classifier(torch.cat(outputs, dim=1)) #在通道维度拼接，从这里看，output是包含通道维度的，是BCHW的
        #if not self.training:
        #    return torch.sigmoid(output)

        outputs.append(output) #outputs包含四张不同尺度的中间特征图，还有一个融合的最终特征图
        outputs = [torch.sigmoid(r) for r in outputs] #sigmoid是最后将每个像素点的边缘强弱映射到0-1之间

        return  x_stages_features, outputs

    def load_from(self, pretrained_path):
            if pretrained_path != "":
                print("pretrained_path:{}".format(pretrained_path))
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                #将预训练权重加载到gpu上
                pretrained_dict = torch.load(pretrained_path, map_location=device)
                print("---start load pretrained modle of edge encoder---")
                pretrained_dict = pretrained_dict['state_dict']
                #根据原始swintransformer的完全权重，适配自己模型的权重
                full_dict = copy.deepcopy(pretrained_dict)
                for k, v in pretrained_dict.items():
                    if "module." in k:
                        current_k =  k[7:]#下采样层数下移
                        # print(current_k)
                        #注意这里的更新，并不是移除之前的，也可以是加入新的
                        full_dict.update({current_k:v})
                        del full_dict[k]
                # 删除有关分类类别的权重
                for k in list(full_dict.keys()):
                    if "head" in k:
                        del full_dict[k]
                msg = self.load_state_dict(full_dict, strict=False) #load_state_dict本身就是一个选择性加载的过程，只加载能匹配到的权重
                # print(msg)
                print('load pretrained modle of edge encoder success')
                print(msg)
            else:
                print("none pretrain edge encoder")
                
    def load_from_sub(self, pretrained_path):
        if pretrained_path != "":
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #将预训练权重加载到gpu上
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            print("---start load pretrained modle of edge encoder---")
            pretrained_dict = pretrained_dict['state_dict']
            #根据原始swintransformer的完全权重，适配自己模型的权重
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                # print(k)
                if "module." in k:
                    current_k =  k[7:]#下采样层数下移
                    # print(current_k)
                    #注意这里的更新，并不是移除之前的，也可以是加入新的
                    full_dict.update({current_k:v})
                    del full_dict[k]
            # 删除有关分类类别的权重
            for k in list(full_dict.keys()):
                if "head" in k:
                    del full_dict[k]
            msg = self.load_state_dict(full_dict, strict=False) #load_state_dict本身就是一个选择性加载的过程，只加载能匹配到的权重
            # print(msg)
            print('load pretrained modle of edge encoder success')
            # print(msg)
        else:
            print("none pretrain edge encoder")
 
class PiDiNetMini(nn.Module):
    def __init__(self, inplane, pdcs, dil=None, sa=False, convert=False):
        super(PiDiNetMini, self).__init__()
        self.sa = sa
        if dil is not None:
            assert isinstance(dil, int), 'dil should be an int'
        self.dil = dil

        self.fuseplanes = []

        self.inplane = inplane

            # self.init_block = Conv2d(pdcs[0], 3, self.inplane, kernel_size=3, padding=1)
            #初始化操作，将原图下采样四倍
        self.init_block = nn.Sequential(
            nn.Conv2d(3, inplane, kernel_size=7, stride=2, padding=3, bias=False),
            nn.InstanceNorm2d(inplane),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        ) # 224 * 224 * 3 -> 112 * 112 * 48  -> 56 * 56 * 48
        block_class = PDCBlock

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block1_1 = block_class(pdcs[0], inplane, self.inplane)
        self.block1_2 = block_class(pdcs[1], self.inplane, self.inplane)
        self.block1_3 = block_class(pdcs[2], self.inplane, self.inplane)
        self.block1_4 = block_class(pdcs[3], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane) # C  56 * 56 * 48 -> 56 * 56 * 96 

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block2_1 = block_class(pdcs[4], inplane, self.inplane, stride=2)
        self.block2_2 = block_class(pdcs[5], self.inplane, self.inplane)
        self.block2_3 = block_class(pdcs[6], self.inplane, self.inplane)
        self.block2_4 = block_class(pdcs[7], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane) # 2C 56 * 56 * 96  -> 28 * 28 * 192
        
        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block3_1 = block_class(pdcs[8], inplane, self.inplane, stride=2)
        self.block3_2 = block_class(pdcs[9], self.inplane, self.inplane)
        self.block3_3 = block_class(pdcs[10], self.inplane, self.inplane)
        self.block3_4 = block_class(pdcs[11], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane) # 4C 28 * 28 * 192 -> 14 * 14 * 384

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block4_1 = block_class(pdcs[12], inplane, self.inplane, stride=2)
        self.block4_2 = block_class(pdcs[13], self.inplane, self.inplane)
        self.block4_3 = block_class(pdcs[14], self.inplane, self.inplane)
        self.block4_4 = block_class(pdcs[15], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane) # 8C 14 * 14 * 384 -> 7 * 7 * 768

        self.conv_reduces = nn.ModuleList()
        if self.sa and self.dil is not None:
            self.attentions = nn.ModuleList()
            self.dilations = nn.ModuleList()
            for i in range(4):
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                self.attentions.append(CSAM(self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        elif self.sa:
            self.attentions = nn.ModuleList()
            for i in range(4):
                self.attentions.append(CSAM(self.fuseplanes[i]))
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))
        elif self.dil is not None:
            self.dilations = nn.ModuleList()
            for i in range(4):
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        else:
            for i in range(4):
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))

        self.classifier = nn.Conv2d(4, 1, kernel_size=1) # has bias
        nn.init.constant_(self.classifier.weight, 0.25)
        nn.init.constant_(self.classifier.bias, 0)

        print('initialization done')

    def get_weights(self):
        conv_weights = []
        bn_weights = []
        relu_weights = []
        for pname, p in self.named_parameters():
            if 'bn' in pname:
                bn_weights.append(p)
            elif 'relu' in pname:
                relu_weights.append(p)
            else:
                conv_weights.append(p)

        return conv_weights, bn_weights, relu_weights

    def forward(self, x):
        H, W = x.size()[2:]

        x = self.init_block(x)

        x1 = self.block1_1(x)
        x1 = self.block1_2(x1)
        x1 = self.block1_3(x1)

        x2 = self.block2_1(x1)
        x2 = self.block2_2(x2)
        x2 = self.block2_3(x2)
        x2 = self.block2_4(x2)

        x3 = self.block3_1(x2)
        x3 = self.block3_2(x3)
        x3 = self.block3_3(x3)
        x3 = self.block3_4(x3)

        x4 = self.block4_1(x3)
        x4 = self.block4_2(x4)
        x4 = self.block4_3(x4)
        x4 = self.block4_4(x4)

        x_stages_features = [x1, x2, x3, x4]
        x_fuses = []
        if self.sa and self.dil is not None:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.attentions[i](self.dilations[i](xi)))
        elif self.sa:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.attentions[i](xi))
        elif self.dil is not None:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.dilations[i](xi))
        else:
            x_fuses = [x1, x2, x3, x4]

        e1 = self.conv_reduces[0](x_fuses[0])
        e1 = F.interpolate(e1, (H, W), mode="bilinear", align_corners=False)

        e2 = self.conv_reduces[1](x_fuses[1])
        e2 = F.interpolate(e2, (H, W), mode="bilinear", align_corners=False)

        e3 = self.conv_reduces[2](x_fuses[2])
        e3 = F.interpolate(e3, (H, W), mode="bilinear", align_corners=False)

        e4 = self.conv_reduces[3](x_fuses[3])
        e4 = F.interpolate(e4, (H, W), mode="bilinear", align_corners=False)

        outputs = [e1, e2, e3, e4]

        output = self.classifier(torch.cat(outputs, dim=1))
        #if not self.training:
        #    return torch.sigmoid(output)

        outputs.append(output)
        outputs = [torch.sigmoid(r) for r in outputs]
        return x_stages_features, outputs
    
    def load_from(self, pretrained_path):
            if pretrained_path != "":
                print("pretrained_path:{}".format(pretrained_path))
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                #将预训练权重加载到gpu上
                pretrained_dict = torch.load(pretrained_path, map_location=device)
                print("---start load pretrained modle of edge encoder---")
                pretrained_dict = pretrained_dict['state_dict']
                #根据原始swintransformer的完全权重，适配自己模型的权重
                full_dict = copy.deepcopy(pretrained_dict)
                for k, v in pretrained_dict.items():
                    if "module." in k:
                        current_k =  k[7:]#下采样层数下移
                        # print(current_k)
                        #注意这里的更新，并不是移除之前的，也可以是加入新的
                        full_dict.update({current_k:v})
                        del full_dict[k]
                # 删除有关分类类别的权重
                for k in list(full_dict.keys()):
                    if "head" in k:
                        del full_dict[k]
                msg = self.load_state_dict(full_dict, strict=False) #load_state_dict本身就是一个选择性加载的过程，只加载能匹配到的权重
                # print(msg)
                print('load pretrained modle of edge encoder success')
            else:
                print("none pretrain edge encoder")    
   
    
def pidinet_tiny(args):
    pdcs = config_model(args.config)
    dil = 8 if args.dil else None
    return PiDiNet(20, pdcs, dil=dil, sa=args.sa)

def pidinet_small(args):
    pdcs = config_model(args.config)
    dil = 12 if args.dil else None
    return PiDiNet(30, pdcs, dil=dil, sa=args.sa)

def pidinet(config,inplane=60, dil=False ,sa=False):
    pdcs = config_model(config)
    dil = 24 if dil else None
    return PiDiNet(inplane, pdcs, dil=dil, sa=sa)

def pidinet_mini(config,inplane=96, dil=False ,sa=False):
    pdcs = config_model(config)
    dil = 24 if dil else None
    return PiDiNetMini(inplane, pdcs, dil=dil, sa=sa)

## convert pidinet to vanilla cnn

def pidinet_tiny_converted(args):
    pdcs = config_model_converted(args.config)
    dil = 8 if args.dil else None
    return PiDiNet(20, pdcs, dil=dil, sa=args.sa, convert=True)

def pidinet_small_converted(args):
    pdcs = config_model_converted(args.config)
    dil = 12 if args.dil else None
    return PiDiNet(30, pdcs, dil=dil, sa=args.sa, convert=True)

def pidinet_converted(args):
    pdcs = config_model_converted(args.config)
    dil = 24 if args.dil else None
    return PiDiNet(60, pdcs, dil=dil, sa=args.sa, convert=True)
