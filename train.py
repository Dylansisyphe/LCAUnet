'''
Descripttion: Created by QiSen Ma
Author: QiSen Ma
Date: 2023-01-15 23:08:29
LastEditTime: 2023-02-04 15:43:27
'''
import yaml
import sys
import time
import argparse
import logging
import os
import random
import numpy as np
import torch
import albumentations as A
import cv2
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from dataset.dataloader import ISIC2018_DataLoader
from dataset.dataloader_kfold import ISIC2018_kfold_DataLoader
from dataset.dataloader_npy import ISIC2017_DataLoader_NPY
from torch.utils.data import DataLoader
from utils import draw_contrast, test_dataset
from configs.config import get_config
from models.edge_module import pidinet, pidinet_mini
from models.LCAUnet import LCAUnet
import torch.nn.functional as F
from losses import BCELoss, DiceLoss, cross_entropy_loss_RCF_Multi_Scale, cross_entropy_loss_RCF
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import AverageMeter
import copy


os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致

swin_config = get_config()
train_config = yaml.load(open('./configs/config_skin.yml'), Loader=yaml.FullLoader)

if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    # print('swin_config.MODEL.NUM_CLASSES:',swin_config.MODEL.NUM_CLASSES)
    data_transform = {
        "train": A.Compose([
                # A.Resize(height=int(swin_config.DATA.IMG_SIZE * 1.143), width=int(swin_config.DATA.IMG_SIZE * 1.143), p=1.0),
                # # # A.CenterCrop(swin_config.DATA.IMG_SIZE,swin_config.DATA.IMG_SIZE),
                # A.RandomResizedCrop(height=swin_config.DATA.IMG_SIZE,width=swin_config.DATA.IMG_SIZE,scale=(0.8, 1.0), ratio=(0.8, 1.25),interpolation=cv2.INTER_LINEAR,p=1.0),
                A.Resize(height=int(swin_config.DATA.IMG_SIZE), width=int(swin_config.DATA.IMG_SIZE), p=1.0),
                A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=15, p=0.7),
                A.Flip(always_apply=False, p=0.7),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit= 30, val_shift_limit=20, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
            ]),
        "validation": A.Compose([
                # A.Resize(height=int(swin_config.DATA.IMG_SIZE * 1.143), width=int(swin_config.DATA.IMG_SIZE * 1.143), p=1.0),
                # A.CenterCrop(swin_config.DATA.IMG_SIZE,swin_config.DATA.IMG_SIZE),
                A.Resize(height=int(swin_config.DATA.IMG_SIZE), width=int(swin_config.DATA.IMG_SIZE), p=1.0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
            ])
    }

    # train_dataset = ISIC2018_DataLoader(data_path=train_config['data_path'],split='train',transform=data_transform['train'])
    train_dataset = ISIC2017_DataLoader_NPY(data_path=train_config['data_path'],split='train',transform=data_transform['train'])
    trainloader = DataLoader(train_dataset, batch_size=train_config['batch_size_tr'], shuffle=True, num_workers=6)

    # valid_dataset = ISIC2018_DataLoader(data_path=train_config['data_path'],split='validation',transform=data_transform['validation'])
    valid_dataset = ISIC2017_DataLoader_NPY(data_path=train_config['data_path'],split='validation',transform=data_transform['validation'])
    validloader = DataLoader(valid_dataset, batch_size=train_config['batch_size_va'], shuffle=True, num_workers=6)

    #----------------edge module-------------------#
    edge_model = pidinet('carv4',  inplane=60, dil=True, sa=True).cuda()
    # edge_model = pidinet_mini('carv4',dil=True, sa=True).cuda()
    edge_model.load_from(train_config['pretrained_edge']) 
    for name, para in edge_model.named_parameters():
        para.requires_grad_(False)
    #----------------------------------------------#


    #仅用config的配置，标准化模型实现
    model = LCAUnet(
                    # img_size=swin_config.DATA.IMG_SIZE,
                    patch_size=swin_config.MODEL.SWIN.PATCH_SIZE,
                    in_chans=swin_config.MODEL.SWIN.IN_CHANS,
                    num_classes=swin_config.MODEL.NUM_CLASSES,
                    embed_dim=swin_config.MODEL.SWIN.EMBED_DIM,
                    depths=swin_config.MODEL.SWIN.DEPTHS,
                    num_heads=swin_config.MODEL.SWIN.NUM_HEADS,
                    window_size=swin_config.MODEL.SWIN.WINDOW_SIZE,
                    mlp_ratio=swin_config.MODEL.SWIN.MLP_RATIO,
                    qkv_bias=swin_config.MODEL.SWIN.QKV_BIAS,
                    qk_scale=swin_config.MODEL.SWIN.QK_SCALE,
                    drop_rate=swin_config.MODEL.DROP_RATE,
                    drop_path_rate=swin_config.MODEL.DROP_PATH_RATE,
                    ape=swin_config.MODEL.SWIN.APE,
                    patch_norm=swin_config.MODEL.SWIN.PATCH_NORM,
                    use_checkpoint=swin_config.TRAIN.USE_CHECKPOINT
                  ).cuda()
    model.load_from(train_config['swin_pretrained_path'])
    if train_config['pretrained'] !='':
        model.load_state_dict(torch.load(train_config['pretrained'])['model_weights'], strict=True)

    ce_loss = BCELoss()
    dice_loss = DiceLoss()
    # bce_loss = nn.BCELoss()
    
    # optimizer = optim.Adam(model.parameters(), lr=float(train_config['lr']), weight_decay=0.0001)
 
    
    optimizer = optim.AdamW(model.parameters(), lr=float(train_config['lr']), weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = 10)

    # 获取当前时间,并将其格式化
    experiment_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time())))
    writer = SummaryWriter('./runs/ISIC2017/' +experiment_time+'/')

    iter_num = 0
    max_epoch = train_config['epochs']
    max_iterations = train_config['epochs'] * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    snapshot_path = './save_models'
    best_performance = 0.0
    best_val_loss  = np.inf
    # 平均数存储与计算器，在展示时用
    losses_ce = AverageMeter();losses_dice = AverageMeter()
    losses_seg = AverageMeter();losses_edge = AverageMeter()
    losses_boundary = AverageMeter()
    losses_total = AverageMeter()
    
    save_dir = './save_models/'+ train_config['experiment_name']
    if not os.path.isdir('./save_models/'+ train_config['experiment_name']):
        os.makedirs(save_dir)
    

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        
        # train one epoch
        model.train()
        # edge_model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch, edge_batch = sampled_batch['image'], sampled_batch['mask'], sampled_batch['edge']
            image_batch, label_batch, edge_batch = image_batch.cuda(), label_batch.cuda(), edge_batch.cuda()
            #使用最后的特征?
            edge_features, edge_outputs = edge_model(image_batch)
            
            outputs_feature, outputs_boundary, outputs_maps = model(image_batch, edge_features)
            
            # print('outputs_feature.shape:',outputs_feature.shape)
            # print('outputs_boundary.shape:',outputs_boundary.shape)
            # print('output_edge.shape:',output_edge.size)
            
            # l_seg
            loss_dice = dice_loss(outputs_feature, label_batch) #对于每个类别进行预测
            # loss_ce = ce_loss(outputs_feature, label_batch[:].long())
            loss_ce = ce_loss(outputs_feature, label_batch)
            loss_seg = train_config['lamda_dice'] * loss_dice + train_config['lamda_ce'] * loss_ce
            
            # print('loss_dice:',loss_dice)
            # print('loss_ce:',loss_ce)
            # l_edge
            loss_edge = cross_entropy_loss_RCF_Multi_Scale(outputs_maps, edge_batch, lmbda=1.1)
            # loss_edge = 0
            
            # l_boundary
            loss_boundary = cross_entropy_loss_RCF(outputs_boundary, edge_batch, lmbda=1.1)
            # loss_boundary = 0
            
            # l_fusion
            loss_fusion = loss_seg + float(train_config['lamda_boundary']) * loss_boundary  + float(train_config['lamda_edge']) * loss_edge
            # backpropagation
            optimizer.zero_grad()
            loss_fusion.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('info/total_loss', loss_fusion, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/loss_seg', loss_seg, iter_num)
            writer.add_scalar('info/loss_edge', loss_edge, iter_num)
            # writer.add_scalar('info/loss_boundary', loss_boundary, iter_num)

            # update metric
            bsz = image_batch.shape[0]
            losses_total.update(loss_fusion, bsz)
            losses_ce.update(loss_ce, bsz)
            losses_dice.update(loss_dice, bsz)
            losses_seg.update(loss_seg, bsz)
            losses_edge.update(loss_edge, bsz)
            # losses_boundary.update(loss_boundary, bsz)
            
            # print loss value
            if i_batch % int(float(train_config['progress_p']) * len(trainloader))==0:
                print('\n----------------------------------------------------------------------')
                print(
                    f"Train: [{epoch_num}][{i_batch + 1}/{len(trainloader)}]\t\t"
                    f"losses_total: {losses_total.val:.3f} ({losses_total.avg:.3f})\n"
                    f"loss_ce: {losses_ce.val:.3f} ({losses_ce.avg:.3f})\t\t"
                    f"losses_dice: {losses_dice.val:.3f} ({losses_dice.avg:.3f})\n"
                    f"losses_seg {losses_seg.val:.3f} ({losses_seg.avg:.3f})\t"
                    f"losses_edge {losses_edge.val:.3f} ({losses_edge.avg:.3f})"
                    # f"loss_boundary: {losses_boundary.val:.3f} ({losses_boundary.avg:.3f})"
                )
                print('----------------------------------------------------------------------')
                sys.stdout.flush()  #清空缓冲区，立即输出打印结果

            # record train effect image
            if iter_num % 20 == 0:
                image = image_batch[0, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                
                # outputs = torch.argmax(torch.softmax(outputs_feature, dim=1), dim=1, keepdim=True)
                outputs = outputs_feature
                # print("outputs[1, ...].shape",outputs[1, ...].shape)
                # print("outputs_boundary[1, ...].shape",outputs_boundary[1, ...].shape)
                # print("label_batch[1, ...].unsqueeze(0)",label_batch[1, ...].unsqueeze(0).shape)
                # print("output_edge[1,4,...]",output_edge[4].shape)
                writer.add_image('train/Prediction', outputs[0, ...] * 50, iter_num)
                writer.add_image('train/PredictionBoundary', outputs_boundary[0, ...] * 50, iter_num)
                labs = label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
                writer.add_image('train/EdgeDetection0', edge_outputs[0][0, ...] * 50, iter_num)
                writer.add_image('train/EdgeDetection1', edge_outputs[1][0, ...] * 50, iter_num)
                writer.add_image('train/EdgeDetection2', edge_outputs[2][0, ...] * 50, iter_num)
                writer.add_image('train/EdgeDetection3', edge_outputs[3][0, ...] * 50, iter_num)
                writer.add_image('train/EdgeDetection4', edge_outputs[4][0, ...] * 50, iter_num)

 
        ## Validation phase
        with torch.no_grad():
            val_loss = 0
            val_loss_ce = 0
            val_loss_dice = 0
            model.eval()
            for itter, sampled_batch in enumerate(validloader):
                image_batch, label_batch = sampled_batch['image'], sampled_batch['mask']
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                edge_features, edge_outputs = edge_model(image_batch)
                outputs_feature, _ , _ = model(image_batch, edge_features)
                # 验证集仅计算l_seg
                loss_ce = ce_loss(outputs_feature, label_batch)
                loss_dice = dice_loss(outputs_feature, label_batch)
                loss_seg = train_config['lamda_dice'] * loss_dice + train_config['lamda_ce'] * loss_ce
                
                val_loss_ce += loss_ce.item()
                val_loss_dice += loss_dice.item()
                val_loss += loss_seg.item()
            print(f' validation on epoch>> {epoch_num} semantic loss>> {(abs(val_loss/(itter+1)))} '
                  f'ce loss>> {(abs(val_loss_ce/(itter+1)))}  dice loss>> {(abs(val_loss_dice/(itter+1)))}')
            mean_val_loss = (val_loss/(itter+1))
            # Check the performance and save the model
            # if (mean_val_loss) < best_val_loss and epoch_num > int(max_epoch / 2) :
            if (mean_val_loss) < best_val_loss :
                print('New best loss, saving...')
                best_val_loss = copy.deepcopy(mean_val_loss)
                state = copy.deepcopy({'model_weights': model.state_dict(), 'val_loss': best_val_loss})
                torch.save(state, save_dir +'/best_epochnum_'+ str(epoch_num) +'_weights_ISIC17.model')

        
                
        # save model per interval
        save_interval = train_config['save_interval']
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        if (epoch_num + 1) % save_interval == 0:
            state = copy.deepcopy({'model_weights': model.state_dict(), 'val_loss': best_val_loss})
            torch.save(state, save_dir +'/interval_save_'+ str(epoch_num) +'_weights_ISIC17.model')    
                
        scheduler.step(mean_val_loss)

    writer.close()

