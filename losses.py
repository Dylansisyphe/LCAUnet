'''
Descripttion: Created by QiSen Ma
Author: QiSen Ma
Date: 2023-01-15 23:12:58
LastEditTime: 2023-02-04 15:29:22
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np






#现在这个函数的问题是，只能计算一个图像的输出?
#怎么修改为让其可以计算多个图像的输出
def cross_entropy_loss_RCF(prediction, labelf, lmbda):
    """
        语义分割的最后的边缘检测约束
    """
    label = labelf.long()
    mask = labelf.clone()#mask是针对每一个像素点的权重
    num_positive = torch.sum(label==1).float()
    num_negative = torch.sum(label==0).float()

    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
    #非边缘像素点的标签为负样本
    mask[label == 0] = lmbda * num_positive / (num_positive + num_negative)
    mask[label == 2] = 0 #忽略弱背景
    cost = nn.BCELoss(weight=mask, reduction='mean')(prediction.squeeze(1), labelf)

    # cost = F.binary_cross_entropy(
    #         prediction.squeeze(1), labelf, weight=mask, reduction='mean')

    return cost

def cross_entropy_loss_RCF_Multi_Scale(predictions, labelf, lmbda):
    """
        边缘网络最后的
    """
    #prediction为tensor格式
    #labelf为numpy格式，可能在数据加载时会被自动转为tensor

    #这些计算确实会对batch有影响？
    #下面这些计算应该针对单张图片而言，还是针对一个batch的图片而言，或者从最后的作用来讲，两者是否有明显区别

    # print('labelf.dtype:',labelf.dtype)
    labelf = labelf.unsqueeze(1)
    label = labelf.long()
    mask = labelf.clone()#mask是针对每一个像素点的权重
    num_positive = torch.sum(label==1).float()
    num_negative = torch.sum(label==0).float()
    # 生成对正样本优化的权重
    # 负样本对应着图像的背景,这里将其较大的比例与正样本相乘，放大正样本的影响
    # print('num_negative:',torch.sum(label==0).float())
    # print('num_positive:',num_positive)
    # print('num_negative / (num_positive + num_negative):',num_negative / (num_positive + num_negative))
    mask[label == 1] = 1 * num_negative / (num_positive + num_negative)
    #非边缘像素点的标签为负样本
    mask[label == 0] = lmbda * num_positive / (num_positive + num_negative)
    mask[label == 2] = 0 #忽略弱背景

    #对不同尺度的特征图均计算损失，然后求和

    #binary_cross_entropy用以评价2分类问题的损失函数，适合于边缘检测这种0-1分类问题
    #该损失函数input的维度(N，*)，其中*表示可以是任何维度，*的维度均属于单个样本。
    #注意该损失函数返回的直接是一个批量的损失，是个标量，而不是每个样本的损失的列表
    costs = 0.0
    for prediction in predictions:
        # print()
        # 生成对正样本优化的权重
        # costs =  costs + nn.BCEWithLogitsLoss(weight=mask, reduction='mean')(prediction.squeeze(1), labelf)
        costs =  costs + nn.BCEWithLogitsLoss(weight=mask, reduction='mean')(prediction, labelf)
    return costs


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1e-5
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss

# def CELoss(inputs, target, num_classes=2,cls_weights=None):
#     n, c, h, w = inputs.size()
#     nt, ht, wt = target.size()
#     if h != ht and w != wt:
#         inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

#     temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
#     temp_target = target.view(-1)

#     CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
#     return CE_loss

# class CELoss(nn.Module):
#     """
#     针对语义分割的CELoss
#     要求输入的数据格式为：(BCHW)
#     """

#     def __init__(self, num_classes=2,cls_weights=None):
#         super(CELoss, self).__init__()
#         self.num_classes = num_classes
#         self.cls_weights = cls_weights

#     def forward(self, inputs, target):
#         n, c, h, w = inputs.size()
#         nt, ht, wt = target.size()
#         if h != ht and w != wt:
#             inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

#         temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
#         temp_target = target.view(-1)
#         # ignore_index：指定标签为什么的时候不参与损失的计算
#         # CE_loss  = nn.CrossEntropyLoss(weight=self.cls_weights, ignore_index= self.num_classes)(temp_inputs, temp_target)
#         CE_loss  = nn.CrossEntropyLoss(weight=self.cls_weights)(temp_inputs, temp_target)
#         return CE_loss


# class DiceLoss(nn.Module):
#     """
#     要求输入的数据格式为：(batch/index of image, height, width, class_map)
#     """

#     def __init__(self, n_classes):
#         super(DiceLoss, self).__init__()
#         self.n_classes = n_classes

#     def _one_hot_encoder(self, input_tensor):
#         """
#             将单通道target通过0-1编码，使得target的通道数和类别数一致
#             使得在计算每个类别的通道所对应的dice时，将target上其他位置的像素点置为0。
#             换句话说，使得其他点的干扰作用相同，视为相同的负样本。
#             #这样做的前提是，target的标号从0开始依次递增，或者需要数据预处理时将target调整成这样。
#             #对于皮肤病数据集，0为背景，1为皮肤病区域。
#         """
#         tensor_list = []
#         for i in range(self.n_classes):
#             temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
#             # tensor_list.append(temp_prob.unsqueeze(1))
#             tensor_list.append(temp_prob.unsqueeze(1))
#         output_tensor = torch.cat(tensor_list, dim=1)
#         return output_tensor.float()

#     def _dice_loss(self, score, target):
#         """
#         对单张图片计算dice loss
#         """
#         target = target.float()
#         smooth = 1e-5
#         intersect = torch.sum(score * target)
#         y_sum = torch.sum(target * target)
#         z_sum = torch.sum(score * score)
#         loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
#         loss = 1 - loss
#         return loss

#     def forward(self, inputs, target, weight=None, softmax=False):

#         if softmax:
#             inputs = torch.softmax(inputs, dim=1)
#         target = self._one_hot_encoder(target)
#         if weight is None:
#             weight = [1] * self.n_classes
#         assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
#         class_wise_dice = []
#         loss = 0.0
#         for i in range(0, self.n_classes):
#             #加上训练约束，使得最后训练出来的类别顺序和多通道target的注释顺序一致
#             dice = self._dice_loss(inputs[:, i], target[:, i])
#             class_wise_dice.append(1.0 - dice.item())
#             loss += dice * weight[i]
#         return loss / self.n_classes