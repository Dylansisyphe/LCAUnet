'''
Descripttion: Created by QiSen Ma
Author: QiSen Ma
Date: 2023-01-26 17:53:54
LastEditTime: 2023-01-29 20:56:48
'''
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange,repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import numpy as np
import copy
from .swin_transformer import PatchEmbed, BasicLayer, PatchMerging
# from .edge_config import config_model
# from .edge_module import EdgeEncoder
from .body_module import LocalCrossAttentionBlock,LocalCrossAttentionBlock_null, LocalCrossAttentionBlock_null2
from .body_module import AMSFF, HeaderFusion, MapReduce, AMSFF_FFM
from .edge_module import pidinet

class LCAUnet_pid(nn.Module):
    r"""

    Args:
        pdcs_configure(str):边缘模块的像素差卷积配置配置
        inplane:边缘模块初始的输出通道个数，该参数对应通道数的变化
        # edge_module(nn.Module): The edge module of LCAUnet.
        ------------------------------------------------------------------------------------
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 2
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        final_upsample(str):final unsample module tpye
    """

    def __init__(self,
                 pdcs_configure='carv4', inplane=60,
                 img_size=224, patch_size=4, in_chans=3, num_classes=2,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        print("LCAUnet initial----num_classes:{}".format(num_classes))

        #------------------------------------edge setting------------------------------------#
        # # edge pdcs setting
        # self.pdcs = config_pdcs(pdcs_configure)
        # # edge dim setting(according to pidnet)
        self.dim_edge = [inplane, 2*inplane, 4*inplane, 4*inplane]
        # # build edge encoder layers
        # self.edge_encoder = EdgeEncoder(inplane,self.pdcs)
        self.edge_model = pidinet('carv4',  inplane=60, dil=True, sa=True)
        #------------------------------------body setting------------------------------------#
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.img_size = img_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth，随机在每一层进行decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build body encoder layers and fusion layers
        self.layers = nn.ModuleList()
        # self.patchMerging_layers = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        patchMerging_layer = None
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample= patchMerging_layer,
                               use_checkpoint=use_checkpoint)
            #将本次的信息传给下一个patchmerging，同时置初始为None,则可以实现将PatchMerging移动到下一个stage开头的作用
            patchMerging_layer = PatchMerging(input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                                patches_resolution[1] // (2 ** i_layer)),
                                               dim=int(embed_dim * 2 ** i_layer),
                                               norm_layer=norm_layer
                                               )
            self.layers.append(layer)

            fusion_layer = LocalCrossAttentionBlock(dim_edge=self.dim_edge[i_layer],
                                                    dim_body=int(embed_dim * 2 ** i_layer),
                                                    input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                                      patches_resolution[1] // (2 ** i_layer)),
                                                    num_heads=num_heads[i_layer],
                                                    window_size=window_size,
                                                    mlp_ratio=self.mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop_ratio=drop_rate, attn_drop_ratio=attn_drop_rate,
                                                    drop_path_ratio=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])][-1],
                                                    norm_layer=norm_layer)
            self.fusion_layers.append(fusion_layer)
        
        
        #针对边缘网络的下采样模块
        self.edge_downsample_layers = nn.ModuleList()
        self.edge_map_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            edge_downsample_layer = nn.Sequential(
                nn.Conv2d(self.dim_edge[i_layer], self.dim_edge[i_layer]*2, kernel_size=2, stride=2, bias=False),
                nn.InstanceNorm2d(self.dim_edge[i_layer]*2),
                nn.ReLU(),
                nn.Conv2d(self.dim_edge[i_layer]*2, self.dim_edge[i_layer], kernel_size=2, stride=2, bias=False),
                nn.InstanceNorm2d(self.dim_edge[i_layer]),
                nn.ReLU(),            
            )
            self.edge_downsample_layers.append(edge_downsample_layer)
            
            edge_map_layer = MapReduce(self.dim_edge[i_layer])
            self.edge_map_layers.append(edge_map_layer)        
            
            
        # encoder最后输出的正则化
        self.norm = norm_layer(self.num_features)

        # build decoder layers，自下而上
        self.layers_up = nn.ModuleList()
        self.num_layers_up = self.num_layers-1
        for i_layer in range(self.num_layers-1):
            layer = AMSFF_FFM(in_channels=int(embed_dim * 2 ** (self.num_layers_up-i_layer)),
                          out_channels=int(embed_dim * 2 ** (self.num_layers_up-1-i_layer)),
                          kernel_size=3,padding=1)
            self.layers_up.append(layer)
        # header
        self.header = HeaderFusion(num_classes=num_classes)

        self.apply(self._init_weights)

    # 虽然使用这里的初始化权重，但实际训练时会将其替换为swintransformer训练好的预训练权重
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


     # Encoder of Edge and Body
    def forward_features(self, x, edge_features):
        # body information
        # 首先进行patch embed操作
        x = self.patch_embed(x)
        # 根据需要判断是否加上绝对路径
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x_downsample = []
        # encoder生成的特征图
        for i_layer in range(self.num_layers):
                # x_downsample一开始是空的列表，所以这里是先加入再处理
                # get body info

                if i_layer > 0:
                    #对fusion得到的特征图进行展平操作
                    B, C, H, W = x.shape
                    x = x.view(B,C,H*W).permute(0,2,1)
                # print('i_layer:',i_layer)
                x = self.layers[i_layer](x)
                # print('i_layer:',i_layer, ' x.shape:', x.shape)
                # print('edge_features[i_layer].shape:',edge_features[i_layer].shape)
                # # edge and body fusion
                x = self.fusion_layers[i_layer](edge_features[i_layer],x)
                # print('i_layer:',i_layer, ' x.shape after fusion', x.shape)
                # 最后一层的正则化
                if i_layer == self.num_layers -1:
                    # x = self.norm(x)
                    pass
                x_downsample.append(x)
        # 返回4个stage的输出
        return x_downsample

    # Dencoder and Skip connection
    def forward_up_features(self, x_downsample):
        #x_downsample 96 -> 192 -> 384 -> 768
        x = x_downsample[self.num_layers_up] #使用最后一层的输出，也就是最下层的采样结果
        for inx, layer_up in enumerate(self.layers_up):
            x = layer_up(x,x_downsample[self.num_layers_up-1-inx])
        return x

    # final unsample and fusion
    def up_x4(self, x,feature_down4):
        x = self.header(x,feature_down4)
        return x
    
    def edge_downsample(self, edge_features):
        edge_downsample_features = []
        edge_downsample_maps = []
        for i_layer in range(self.num_layers):
            # print('edge_features[',i_layer,'].shape: ',edge_features[i_layer].shape)
            edge_downsample_feature = self.edge_downsample_layers[i_layer](edge_features[i_layer])
            edge_downsample_map = self.edge_map_layers[i_layer](edge_downsample_feature)
            edge_downsample_map = F.interpolate(edge_downsample_map, (self.img_size, self.img_size), mode="bilinear", align_corners=False)
            edge_downsample_features.append(edge_downsample_feature)

            
            edge_downsample_maps.append(edge_downsample_map)
            # print('edge_features[',i_layer,']','.shape: ', edge_features[i_layer].shape)
        return edge_downsample_features, edge_downsample_maps

    def forward(self, x):
        edge_features, edge_result= self.edge_model(x)
        
        original_feature = x
        # edge information
        # edge_features, edge_result= self.edge_encoder(x)
        # edge_features
        
        edge_downsample_features, edge_downsample_maps = self.edge_downsample(edge_features)
        # encoder downsample
        x_downsample = self.forward_features(x,edge_downsample_features)
        # decoder unsample
        feature_down4 = self.forward_up_features(x_downsample)
        # final unsample and fusion
        feature_result, boundary_result = self.up_x4(original_feature,feature_down4)
        #feature是正常的区域特征，boundary是根据区域推导出的边界，edge是边缘检测网络的输出
        return feature_result ,boundary_result, edge_downsample_maps

    def load_from(self, pretrained_path, pretrained_edge):
        #----------------------------- load body weight -----------------------#
        if pretrained_path != "":
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #将预训练权重加载到gpu上
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            print("---start load pretrained modle of swin encoder---")
            pretrained_dict = pretrained_dict['model']
            #根据原始swintransformer的完全权重，适配自己模型的权重
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "downsample." in k:
                    current_layer_num = int(k[7:8]) #??这里k[7:8]指的就是索引编号，这里通过相减，起到上采样倒置的作用
                    current_k = "layers." + str(current_layer_num+1) + k[8:] #下采样层数下移
                    # print(current_k)
                    #注意这里的更新，并不是移除之前的，也可以是加入新的
                    full_dict.update({current_k:v})
                    del full_dict[k]
            # for k, v in pretrained_dict.items():
            #     if "downsample" in k:
            #         current_layer_num = int(k[7:8]) #??这里k[7:8]指的就是索引编号，这里通过相减，起到上采样倒置的作用
            #         current_k = "layers." + str(current_layer_num+1) + k[8:] #下采样层数下移
            #         #注意这里的更新，并不是移除之前的，也可以是加入新的
            #         pretrained_dict.update({current_k:v})
            #         del pretrained_dict[k]
            # 删除有关分类类别的权重
            for k in list(full_dict.keys()):
                if "head" in k:
                    del full_dict[k]

            msg = self.load_state_dict(full_dict, strict=False) #load_state_dict本身就是一个选择性加载的过程，只加载能匹配到的权重
            print('load pretrained modle of swin encoder success')
        else:
            print("none pretrain swin encoder")

        #----------------------------- load edge weight -----------------------#
        self.edge_model.load_from_sub(pretrained_edge) 
        # for name, para in self.edge_model.named_parameters():
        #     para.requires_grad_(False)
    
    
    # 想要方法不被编译？
    # 允许模型以python函数的形式保存不被TorchScript兼容的代码，适用于需要在模型（nn.Module）外部调用的方法
    # 猜想可能是因为
    # 以下两个函数来自swin-transformer官方库，在官方库的优化器配置中被使用，用来特定设置不使用权重衰退的属性。
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}