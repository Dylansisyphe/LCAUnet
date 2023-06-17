'''
Descripttion: Created by QiSen Ma
Author: QiSen Ma
Date: 2023-01-26 17:53:54
LastEditTime: 2023-01-30 14:50:43
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
from torch.nn.utils.spectral_norm import spectral_norm
from .body_ops import window_partition, window_reverse, Mlp

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class HeaderFusion(nn.Module):
    """
        最后的特征图融合与结果输出模块
        num_classes: 最后判别的类别, 默认为2
    """
    def __init__(self,num_classes=2):
        super(HeaderFusion,self).__init__()
        #对原始特征图的两次采样，自上而下
        self.initconv  = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=3, padding=1 ,bias=False),
            nn.InstanceNorm2d(48),
            nn.ReLU(),
        )
        
        self.downconv = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(48),
            nn.ReLU(),
        )
 
        self.amsff1 = AMSFF(in_channels=96,out_channels=48) #down2用
        self.amsff2 = AMSFF(in_channels=48,out_channels=48)
        
        #头部输出
        self.finalconv = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=num_classes, kernel_size=1 ,bias=False),
        )

        
        #边缘输出
        self.maxpool = nn.MaxPool2d(3, stride=1,padding=1)

    def forward(self, x, feature_down4):
        #原始特征图的两次采样
        x_init = self.initconv(x) 
        x_down2 = self.downconv(x_init)
        #56*56*96 -> 112*112*96
        x_feature_d2 = self.amsff1(feature_down4,x_down2)

        x_feature = self.amsff2(x_feature_d2,x_init)     
        
        # #原始图像尺寸的拼接  224*224*48 ->  224*224*96
        # x_fusion = torch.cat([x_init, x_feature], 1)
        # #最后输出  224*224*96 -> 224*224*48 -> 224*224*num_classes
        # x_final = self.finalconv(x_fusion)
        x_final = torch.sigmoid(self.finalconv(x_feature))
        
        # print('x_final.shape:',x_final.shape)
        # x_region = torch.argmax(torch.softmax(x_final, dim=1), dim=1).float()
        # print('self.maxpool(x_region).shape:',self.maxpool(x_region).shape)
        x_boundary = self.maxpool(x_final) - x_final
        return x_final, x_boundary

class AMSFF_init(nn.Module):
    """Adaptively Multi-Scale Feature Fusion.
        包含了底尺度特征图的残差上采样模块

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        kernel_size (int): Kernel size in convolutions. Default: 3.
        padding (int): Padding in convolutions. Default: 1.
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1):
        super(AMSFF, self).__init__()
        # self.initconv = nn.Sequential(
        #     nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0),
        #     nn.ReLU()
        # )
        # self.resconv = nn.Sequential(
        #     nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding),
        #     nn.ReLU(),
        # )
        # self.convup = nn.Sequential(
        #     # nn.ConvTranspose2d(out_channel, out_channel, kernel_size=4, stride=2, padding=1),
        #     # nn.ReLU(),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding),
        #     nn.ReLU(),
        # )

        self.initconv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)),
            nn.LeakyReLU(0.02, True)
        )
        self.resconv = nn.Sequential(
            spectral_norm(nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)),
            nn.LeakyReLU(0.02, True),
            spectral_norm(nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)),
            nn.LeakyReLU(0.02, True)
        )
        self.convup = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            spectral_norm(nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)),
            nn.LeakyReLU(0.2, True),
        )

        # for FFM scale and shift
        self.scale_block = nn.Sequential(
            spectral_norm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)))
        self.shift_block = nn.Sequential(
            spectral_norm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)), nn.Sigmoid())

    def forward(self, x, updated_feat):
        # x和updated_feat的H与W需要一致
        # updated_feat 属于低分辨率，更高尺度的语义信息,对细粒度信息进行调制

        # 前处理。使x的通道数与out_channel一致
        x = self.initconv(x)
        #残差卷积并进行upsample
        out = self.convup(self.resconv(x) + x)
        # SFT
        scale = self.scale_block(updated_feat)
        shift = self.shift_block(updated_feat)
        out = (out * scale + shift) + out #按位运算，并加上跳跃连接

        return out


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        # print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out


class ResBlock_CBAM(nn.Module):
    def __init__(self,in_places, places, stride=1):
        super(ResBlock_CBAM,self).__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.InstanceNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm2d(places),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(channel=places)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        # print(x.shape)
        out = self.cbam(out)

        out += residual
        out = self.relu(out)
        return out

class ResBlock_ChannelAttention(nn.Module):
    def __init__(self,in_places, places, stride=1):
        super(ResBlock_ChannelAttention,self).__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.InstanceNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm2d(places),
            nn.ReLU(inplace=True),
        )
        self.channel_attention = ChannelAttentionModule(channel=places)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        # print(x.shape)
        out = self.channel_attention(out) * x

        out += residual
        out = self.relu(out)
        return out


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
    

class AMSFF(nn.Module):
    """Adaptively Multi-Scale Feature Fusion.
        包含了底尺度特征图的残差上采样模块

    Args:
        in_channel (int): Number of input channels. #底层特征图的维度
        out_channel (int): Number of output channels.#较高层特征图的维度
        kernel_size (int): Kernel size in convolutions. Default: 3.
        padding (int): Padding in convolutions. Default: 1.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(AMSFF, self).__init__()
        self.resblock1 = ResBlock(in_channels=out_channels, out_channels=out_channels) #上层特征图初始用
        self.resblock2 = ResBlock(in_channels=in_channels, out_channels=in_channels) #底层特征图用初始用
        self.resblock3 = ResBlock(in_channels=out_channels*2, out_channels=out_channels) #融合特征图用
        # self.resblock_CBAM = ResBlock_CBAM(in_places=out_channels, places=out_channels)
        # self.resblock_channel_attention = ResBlock_ChannelAttention(in_places=out_channels, places=out_channels)
        self.convup = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels)
            # nn.ReLU(),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding),
            # nn.ReLU(),
        )

    def forward(self, x, updated_feat):
        # x是底层特征图
        # updated_feat 是高层特征图
        updated_feat = self.resblock1(updated_feat)
        x = self.resblock2(x)
        x = self.convup(x)
        x_fusion =  torch.cat([x, updated_feat], dim=1)
        out = self.resblock3(x_fusion)
        # out = self.resblock_CBAM(out)
        # out = self.resblock_channel_attention(out)
        return out


class AMSFF_FFM(nn.Module):
    """Adaptively Multi-Scale Feature Fusion.
        包含了底尺度特征图的残差上采样模块

    Args:
        in_channel (int): Number of input channels. #底层特征图的维度
        out_channel (int): Number of output channels.#较高层特征图的维度
        kernel_size (int): Kernel size in convolutions. Default: 3.
        padding (int): Padding in convolutions. Default: 1.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(AMSFF_FFM, self).__init__()
        self.resblock1 = ResBlock(in_channels=out_channels, out_channels=out_channels) #上层特征图初始用
        self.resblock2 = ResBlock(in_channels=in_channels, out_channels=in_channels) #底层特征图用初始用
        self.resblock3 = ResBlock(in_channels=out_channels * 2, out_channels=out_channels) #融合特征图用
        # self.resblock_CBAM = ResBlock_CBAM(in_places=out_channels, places=out_channels)
        # self.resblock_channel_attention = ResBlock_ChannelAttention(in_places=out_channels, places=out_channels)
        self.convup = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels)
            # nn.ReLU(),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding),
            # nn.ReLU(),
        )
        
        # for FFM scale and shift
        self.scale_block = nn.Sequential(
            spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1,bias=False)), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1,bias=False)))
        self.shift_block = nn.Sequential(
            spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1,bias=False)), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1,bias=False)), nn.Sigmoid())

    def forward(self, x, updated_feat):
        # x是底层特征图
        # updated_feat 是高层特征图
        updated_feat = self.resblock1(updated_feat)
        x = self.resblock2(x)
        x = self.convup(x)
        
        # SFT
        scale = self.scale_block(x)
        shift = self.shift_block(x)
        out_aug = (updated_feat * scale + shift) #按位运算
        
        x_fusion =  torch.cat([out_aug, updated_feat], dim=1)
        out = self.resblock3(x_fusion) 
        
        return out

class ResBlock(nn.Module):
    """Adaptively Multi-Scale Feature Fusion.
        包含了底尺度特征图的残差上采样模块

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        kernel_size (int): Kernel size in convolutions. Default: 3.
        padding (int): Padding in convolutions. Default: 1.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1,bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.02)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3, padding=1,bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels)      

        self.downsample = in_channels != out_channels
        if self.downsample:
            self.conv3 = nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1,bias=False)
            self.norm3 = nn.InstanceNorm2d(out_channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        
        out += residual
        out = self.lrelu(out)
        return out




class WindowCrossAttention(nn.Module):
    r""" Window based multi-head cross attention (W-MCA) module with relative position bias.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads. #参考同stage的swintransformer的配置
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop_ratio (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop_ratio (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index) #实际注册时会加上层级信息，所以不会重复

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_edge, x_body):
        """
        Args:
            x_edge: input edge features with shape of (num_windows*B, N, C)
            x_body: input body features with shape of (num_windows*B, N, C)
        """
        B_, N, C = x_body.shape
        #使用x_edge作为query，x_body作为key和value
        q = self.q(x_edge).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x_body).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q = self.q(x_body).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # kv = self.kv(x_edge).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        #进行qk的注意力运算
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        #加入相对位置偏差
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        #softmax处理
        attn = self.softmax(attn)
        #drop处理
        attn = self.attn_drop(attn)

        #进行kqv的注意力运算
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C) #输出shape同输入shape: num_windows*B, N, C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        #Set the extra representation of the module
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

class LocalCrossAttentionBlock(nn.Module):
    """
        一个完整的transformer block, 可在这个基础上进行处理和修改
    args:
        dim_edge (int): Number of edge input channels.
        dim_body (int): Number of body input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (list): Window size for height and weight.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_ratio (float, optional): Dropout rate. Default: 0.0
        attn_drop_ratio (float, optional): Attention dropout rate. Default: 0.0
        drop_path_ratio (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    return: [B,H,W,C] feature information
    """
    def __init__(self,
                 dim_edge,
                 dim_body,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(LocalCrossAttentionBlock, self).__init__()

        self.input_resolution = input_resolution
        self.window_size = window_size

        self.norm_edge = norm_layer([dim_edge, input_resolution[0],input_resolution[1]]) #对于单张图片进行layernorm，维度为C,H,W
        self.conv = nn.Conv2d(dim_edge, dim_body, kernel_size=1, padding=0, bias=False)
        self.norm1 = norm_layer(dim_body)
        self.attn = WindowCrossAttention(dim_body, num_heads=num_heads, window_size=[window_size,window_size], qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_body)
        mlp_hidden_dim = int(dim_body * mlp_ratio)
        self.mlp = Mlp(in_features=dim_body, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x_edge, x_body):
        x_body_cache = x_body.clone()
        H, W = self.input_resolution
        #[B, num_patches, embed_dim]
        B, L, C = x_body.shape
        #norm处理,norm是有存储的weight和bias的
        x_edge = self.norm_edge(x_edge)
        x_body = self.norm1(x_body).view(B, H, W, C)#BLC -> BHWC，方便进行window partition 
        #edge的通道维度更改
        x_edge = self.conv(x_edge).permute(0,2,3,1) #BCHW -> BHWC，方便进行window partition 

        #其实multihead_attn这一步就完成了cross attention，其他的部分可以说是参考transfomer block的设计让其更合理
        #window partition
        x_edge_windows = window_partition(x_edge, self.window_size)  # nW*B, window_size, window_size, C
        x_edge_windows = x_edge_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        x_body_windows = window_partition(x_body, self.window_size)  # nW*B, window_size, window_size, C
        x_body_windows = x_body_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        #window cross attention
        attn_windows = self.attn(x_edge_windows, x_body_windows)  # nW*B, window_size*window_size, C
        #window reverse
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x_attn = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        x_attn = x_attn.view(B, H * W, C)

        #加上跳跃连接,初始为body的特征
        x = x_body_cache + self.drop_path(x_attn)
        #FeedForward NetWork
        x = x + self.drop_path(self.mlp(self.norm2(x)))#[B, num_patches, embed_dim]
        #后续不再进行注意力处理，为了方便后续的特征图的卷积处理，这里返回的为[B,C,H,W]的类型
        x = x.reshape(B, H, W, C).permute(0,3,1,2)
        return x

class LocalCrossAttentionBlock_null(nn.Module):
    """
        一个完整的transformer block, 可在这个基础上进行处理和修改
    args:
        dim_edge (int): Number of edge input channels.
        dim_body (int): Number of body input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (list): Window size for height and weight.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_ratio (float, optional): Dropout rate. Default: 0.0
        attn_drop_ratio (float, optional): Attention dropout rate. Default: 0.0
        drop_path_ratio (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    return: [B,H,W,C] feature information
    """
    def __init__(self,
                 dim_edge,
                 dim_body,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(LocalCrossAttentionBlock_null, self).__init__()

        self.input_resolution = input_resolution
        self.window_size = window_size

        self.norm_edge = norm_layer([dim_edge, input_resolution[0],input_resolution[1]]) #对于单张图片进行layernorm，维度为C,H,W
        self.conv = nn.Conv2d(dim_edge, dim_body, kernel_size=1, padding=0, bias=False)
        self.norm1 = norm_layer(dim_body)
        
        self.fusion_block = ResBlock(in_channels=dim_body*2, out_channels=dim_body) #融合特征图用


    def forward(self, x_edge, x_body):
        x_body_cache = x_body.clone()
        H, W = self.input_resolution
        #[B, num_patches, embed_dim]
        B, L, C = x_body.shape
        #norm处理,norm是有存储的weight和bias的
        x_edge = self.norm_edge(x_edge)
        x_body = self.norm1(x_body).view(B, H, W, C).permute(0,3,1,2)#BLC -> BHWC-> BCHW
        #edge的通道维度更改
        x_edge = self.conv(x_edge)

        x_fusion =  torch.cat([x_body, x_edge], dim=1)
        out = self.fusion_block(x_fusion)
        
        return out


class LocalCrossAttentionBlock_null2(nn.Module):
    """
        一个完整的transformer block, 可在这个基础上进行处理和修改
    args:
        dim_edge (int): Number of edge input channels.
        dim_body (int): Number of body input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (list): Window size for height and weight.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_ratio (float, optional): Dropout rate. Default: 0.0
        attn_drop_ratio (float, optional): Attention dropout rate. Default: 0.0
        drop_path_ratio (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    return: [B,H,W,C] feature information
    """
    def __init__(self,
                 dim_edge,
                 dim_body,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(LocalCrossAttentionBlock_null2, self).__init__()

        self.input_resolution = input_resolution
        self.window_size = window_size

        self.norm_edge = norm_layer([dim_edge, input_resolution[0],input_resolution[1]]) #对于单张图片进行layernorm，维度为C,H,W
        self.conv = nn.Conv2d(dim_edge, dim_body, kernel_size=1, padding=0, bias=False)
        self.norm1 = norm_layer(dim_body)
        

    def forward(self, x_edge, x_body):
        x_body_cache = x_body.clone()
        H, W = self.input_resolution
        #[B, num_patches, embed_dim]
        B, L, C = x_body.shape
        #norm处理,norm是有存储的weight和bias的
        x_edge = self.norm_edge(x_edge)
        x_body = self.norm1(x_body).view(B, H, W, C).permute(0,3,1,2)#BLC -> BHWC-> BCHW
        #edge的通道维度更改
        x_edge = self.conv(x_edge)

        x_fusion =  x_body + x_edge
        
        return x_fusion

