from __future__ import division
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataloader import ISIC2018_DataLoader
import glob
import numpy as np
import copy
import yaml
from tqdm import tqdm
from dataset.dataloader import ISIC2018_DataLoader
from dataset.dataloader_npy import ISIC2017_DataLoader_NPY
from models.LCAUnet import LCAUnet
from sklearn.metrics import f1_score, confusion_matrix
from matplotlib import pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes, binary_opening
from configs.config import get_config
import albumentations as A
import argparse
from configs.config import get_config
from models.edge_module import pidinet
## Hyper parameters
swin_config = get_config()
test_config = yaml.load(open('./configs/config_skin.yml'), Loader=yaml.FullLoader)



device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_transform = {
    "test": A.Compose([
            A.Resize(height=int(swin_config.DATA.IMG_SIZE), width=int(swin_config.DATA.IMG_SIZE), p=1.0),#一定要记得把照片缩放为合适的大小
            # A.Flip(always_apply=False, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
        ])
}

# test_dataset = ISIC2018_DataLoader(data_path=test_config['data_path'],split='test',transform=data_transform['test'])
test_dataset = ISIC2017_DataLoader_NPY(data_path=test_config['data_path'],split='test',transform=data_transform['test'])
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=6, pin_memory=True)

#仅用config的配置，标准化模型实现
model = LCAUnet(pdcs_configure='carv4', inplane=60,
                img_size=swin_config.DATA.IMG_SIZE,
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
                ).to(device)

if test_config['saved_model'] != "":
    print("load from : ",test_config['saved_model'])
    model.load_state_dict(torch.load(test_config['saved_model'], map_location='cpu')['model_weights'])

#----------------edge module-------------------#
edge_model = pidinet('carv4',dil=True, sa=True).cuda()
edge_model.load_from(test_config['pretrained_edge'])
# if train_config['pretrained_edge'] !='':
    
#     edge_model.load_state_dict(torch.load(train_config['pretrained_edge'])['state_dict'], strict=True)

for name, para in edge_model.named_parameters():
    para.requires_grad_(False)
#----------------------------------------------#

def evaluate_dice(model):
    print('Quntitative performance')
    predictions = []
    gt = []

    with torch.no_grad():
        val_loss = 0
        model.eval()
        for i_batch, sampled_batch in tqdm(enumerate(test_loader)):
            img = sampled_batch['image'].to(device, dtype=torch.float)
                        #使用最后的特征?
            edge_features, edge_outputs = edge_model(img)
            
            # outputs_feature, outputs_boundary = model(img, edge_features)
        
            msk_pred, _ , _  = model(img, edge_features)
            mask = sampled_batch['mask']
            gt.append(mask.numpy()[0]) #注意mask只有三个维度,BHW
            #msk_pred的预测单通道转换
            # print('before mask_pred.shape:',msk_pred.shape)
            #对BCHW格式的图像,先选择每一点的分类，再挑选第一张图片（其实就一张图片）
            # msk_pred = torch.argmax(torch.softmax(msk_pred, dim=1), dim=1).squeeze(0)
            # msk_pred = torch.softmax(msk_pred, dim=1).squeeze(0)
            msk_pred = msk_pred.cpu().detach().numpy()
            #这里通过对单通道建立阈值来进行0-1像素的分类，CHW，第一个通道是背景概率，第二个通道是mask概率
            msk_pred  = np.where(msk_pred[1]>=0.43, 1, 0) 
            msk_pred = np.array(msk_pred)
            msk_pred = binary_opening(msk_pred, structure=np.ones((6,6))).astype(msk_pred.dtype)
            msk_pred = binary_fill_holes(msk_pred, structure=np.ones((6,6))).astype(msk_pred.dtype)
            predictions.append(msk_pred)

    predictions = np.array(predictions)
    gt = np.array(gt)

    y_scores = predictions.reshape(-1)
    y_true   = gt.reshape(-1)

    y_scores2 = np.where(y_scores>0.47, 1, 0)
    y_true2   = np.where(y_true>0.5, 1, 0)

    #F1 score
    F1_score = f1_score(y_true2, y_scores2, labels=None, average='binary', sample_weight=None)
    print ("\nF1 score (F-measure) or DSC: " +str(F1_score))   
    return F1_score


# ## Quntitative performance
def evaluate():
    print('Quntitative performance')
    predictions = []
    gt = []

    with torch.no_grad():
        val_loss = 0
        model.eval()
        for i_batch, sampled_batch in tqdm(enumerate(test_loader)):
            img = sampled_batch['image'].to(device, dtype=torch.float)
            mask = sampled_batch['mask']
                        #使用最后的特征?
            edge_features, edge_outputs = edge_model(img)
            
            # outputs_feature, outputs_boundary = model(img, edge_features)
        
            msk_pred, _ , _ = model(img, edge_features)
            gt.append(mask.numpy()[0]) #注意mask只有三个维度,BHW
            #msk_pred的预测单通道转换
            # print('before mask_pred.shape:',msk_pred.shape)
            #对BCHW格式的图像,先选择每一点的分类，再挑选第一张图片（其实就一张图片）
            # msk_pred = torch.argmax(torch.softmax(msk_pred, dim=1), dim=1).squeeze(0)
            # msk_pred = torch.softmax(msk_pred, dim=1).squeeze(0)
            msk_pred = msk_pred.cpu().detach().numpy()[0, 0]
            #这里通过对单通道建立阈值来进行0-1像素的分类，CHW，第一个通道是背景概率，第二个通道是mask概率
            # msk_pred  = np.where(msk_pred[1]>=0.43, 1, 0) 
            msk_pred  = np.where(msk_pred>=0.43, 1, 0)

        
            msk_pred = np.array(msk_pred)
            msk_pred = binary_opening(msk_pred, structure=np.ones((6,6))).astype(msk_pred.dtype)
            msk_pred = binary_fill_holes(msk_pred, structure=np.ones((6,6))).astype(msk_pred.dtype)
            predictions.append(msk_pred)

    predictions = np.array(predictions)
    gt = np.array(gt)

    y_scores = predictions.reshape(-1)
    y_true   = gt.reshape(-1)

    y_scores2 = np.where(y_scores>0.47, 1, 0)
    y_true2   = np.where(y_true>0.5, 1, 0)

    #F1 score
    F1_score = f1_score(y_true2, y_scores2, labels=None, average='binary', sample_weight=None)
    print ("\nF1 score (F-measure) or DSC: " +str(F1_score))
    confusion = confusion_matrix(np.int32(y_true), y_scores2)
    print (confusion)
    sensitivity = 0
    if float(confusion[1,1]+confusion[1,0])!=0:
        sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
    print ("Sensitivity: " +str(sensitivity))
    specificity = 0
    if float(confusion[0,0]+confusion[0,1])!=0:
        specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
    print ("Specificity or Recall: " +str(specificity))
    accuracy = 0
    if float(np.sum(confusion))!=0:
        accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
    print ("Accuracy: " +str(accuracy))




# ## Visualization section
def save_sample(img, msk, msk_pred, th=0.3, name='', title=''):   
    img2 = img.detach().cpu().numpy()[0]
    img2 = np.einsum('kij->ijk', img2)
    msk2 = msk.detach().cpu().numpy()[0]
    print(msk2.shape)
    mskp = msk_pred.detach().cpu().numpy()[0]
    msk2 = np.where(msk2>0.5, 1., 0)
    mskp = np.where(mskp>=th, 1., 0)

    plt.figure(figsize=(5,2))
    plt.suptitle(title, fontsize=15)

    plt.subplot(1,3,1)
    plt.imshow(img2)
    plt.title('Normalized image', y=-0.3)
    plt.axis('off')
    
    plt.subplot(1,3,2)
    plt.imshow(msk2*255, cmap= 'gray')
    plt.title('Ground Truth', y=-0.3)
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(mskp*255, cmap = 'gray')
    plt.title('Prediction', y=-0.3)
    plt.axis('off')

    plt.savefig('./results/'+name+'.png')
    
def draw(num_pictures=5):
    print('Visualization')
    gt = []
    N = num_pictures ## Number of samples to visualize
    with torch.no_grad():
        val_loss = 0
        model.eval()
        for itter, sampled_batch in tqdm(enumerate(test_loader)):
            img = sampled_batch['image'].to(device, dtype=torch.float)
            mask = sampled_batch['mask']
            case_name = sampled_batch['case_name'][0]
            msk_pred, _ = model(img)
            msk_pred = msk_pred.argmax(dim=1)
            gt.append(mask.numpy())
            save_sample(img, mask, msk_pred, th=0.5, name=str(itter+1) ,title=case_name)
            if itter+1==N:
                break


            
parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str, default='evaluate', help='evaluate or draw')
args = parser.parse_args()

if __name__ == "__main__":
    if args.opt == 'evaluate':
        evaluate()
    elif args.opt == 'draw':
        draw()
    else:
        print('Please input correct operation')