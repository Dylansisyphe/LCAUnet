# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:15:43 2019
@author: Reza Azad
"""
from __future__ import division
import numpy as np
import scipy.io as sio
import scipy.misc as sc
import glob
from tqdm import tqdm
import cv2

# Parameters
height = 224
width  = 224
channels = 3

############################################################# Prepare ISIC 2017 data set #################################################
Dataset_add = '/root/autodl-tmp/datasets/ISIC2017/'
Save_add = '/root/autodl-tmp/datasets/ISIC2017/'

Train_add = 'ISIC-2017_Training_Data/'
Train_label_add = 'ISIC-2017_Training_Part1_GroundTruth/'

Valid_add = 'ISIC-2017_Validation_Data/'
Valid_label_add = 'ISIC-2017_Validation_Part1_GroundTruth/'

Test_add = 'ISIC-2017_Test_v2_Data/'
Test_label_add = 'ISIC-2017_Test_v2_Part1_GroundTruth/'


def gen_grayworld_image(image):
    r,g,b = np.split(image,3,axis=2)
 
    r_avg = np.mean(r)
    g_avg = np.mean(g)
    b_avg = np.mean(b) #可能就一个元素
    avg = (b_avg+g_avg+r_avg)/3

    r_k = avg/r_avg
    g_k = avg/g_avg
    b_k = avg/b_avg
    
    r = np.clip(r*r_k,0,255)
    g = np.clip(g*g_k,0,255)
    b = np.clip(b*b_k,0,255)
     
    image = cv2.merge([r,g,b]).astype(np.uint8)
    return image
############################################################# Prepare ISIC 2017 train dataset###########################################
Tr_list = glob.glob(Dataset_add+ Train_add+'/*.jpg')
print('训练集图片的数量为:',len(Tr_list))
# It contains 2000 training samples
Data_train_2017    = np.zeros([len(Tr_list), height, width, channels])
Label_train_2017   = np.zeros([len(Tr_list), height, width])

print('Reading ISIC 2017 train data')
for idx in tqdm(range(len(Tr_list))):
    img = sc.imread(Tr_list[idx])
    
    img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode = 'RGB'))
    img = gen_grayworld_image(img)
    Data_train_2017[idx, :,:,:] = img
    
    b = Tr_list[idx]    
    a = b[0:len(Dataset_add)]
    b = b[len(b)-16: len(b)-4] 
    add = (a+ Train_label_add + b +'_segmentation.png')    
    img2 = sc.imread(add)
    img2 = np.double(sc.imresize(img2, [height, width], interp='bilinear'))
    Label_train_2017[idx, :,:] = img2    
         
print('Reading ISIC 2017 train dataset finished')

np.save(Save_add + 'data_train_gw', Data_train_2017)
np.save(Save_add + 'mask_train_gw' , Label_train_2017)



############################################################# Prepare ISIC 2017 valid dataset###########################################
Valid_list = glob.glob(Dataset_add+ Valid_add+'/*.jpg')
print('验证集图片的数量为:',len(Valid_list))
# It contains 150 valid samples
Data_valid_2017    = np.zeros([len(Valid_list), height, width, channels])
Label_valid_2017   = np.zeros([len(Valid_list), height, width])

print('Reading ISIC 2017 valid data')
for idx in tqdm(range(len(Valid_list))):
    img = sc.imread(Valid_list[idx])
    
    img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode = 'RGB'))
    img = gen_grayworld_image(img)
    Data_valid_2017[idx, :,:,:] = img

    
    b = Valid_list[idx]    
    a = b[0:len(Dataset_add)]
    b = b[len(b)-16: len(b)-4] 
    add = (a+ Valid_label_add + b +'_segmentation.png')    
    img2 = sc.imread(add)
    img2 = np.double(sc.imresize(img2, [height, width], interp='bilinear'))
    Label_valid_2017[idx, :,:] = img2    
         
print('Reading ISIC 2017 valid dataset finished')

np.save(Save_add + 'data_val_gw', Data_valid_2017)
np.save(Save_add + 'mask_val_gw' , Label_valid_2017)


############################################################# Prepare ISIC 2017 test dataset###########################################
Test_list = glob.glob(Dataset_add+ Test_add+'/*.jpg')
print('测试集图片的数量为:',len(Test_list))
# It contains 600 test samples
Data_test_2017    = np.zeros([len(Test_list), height, width, channels])
Label_test_2017   = np.zeros([len(Test_list), height, width])

print('Reading ISIC 2017 test data')
for idx in tqdm(range(len(Test_list))):
    img = sc.imread(Test_list[idx])
    
    img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode = 'RGB'))
    img = gen_grayworld_image(img)
    Data_test_2017[idx, :,:,:] = img

    
    b = Test_list[idx]    
    a = b[0:len(Dataset_add)]
    b = b[len(b)-16: len(b)-4] 
    add = (a+ Test_label_add + b +'_segmentation.png')    
    img2 = sc.imread(add)
    img2 = np.double(sc.imresize(img2, [height, width], interp='bilinear'))
    Label_test_2017[idx, :,:] = img2    
         
print('Reading ISIC 2017 test dataset finished')

np.save(Save_add + 'data_test_gw', Data_test_2017)
np.save(Save_add + 'mask_test_gw' , Label_test_2017)

