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
import random
from tqdm import tqdm
import cv2
#from sklearn.model_selection import train_test_split

# Parameters
height = 224
width  = 224
channels = 3

############################################################# Prepare ph2 data set #################################################
Dataset_add = '/root/autodl-tmp/datasets/PH2/PH2 Dataset images/'
Save_add = '/root/autodl-tmp/datasets/PH2/'
data_suffix = '_Dermoscopic_Image'
mask_suffix = '_lesion'


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


Data_list = glob.glob(Dataset_add+'/*')
print(Data_list)
print(len(Data_list))

Data_train    = np.zeros([200, height, width, channels])
Label_train   = np.zeros([200, height, width])

print('Reading Ph2')

random.shuffle(Data_list)

for idx in tqdm(range(len(Data_list))):
    name = Data_list[idx].split(sep='/')[-1]
    # print('name',name)
    # print(Data_list[idx])
    #data
    img = sc.imread(Data_list[idx]+'/' + name + data_suffix + '/' + name + '.bmp')
    img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode = 'RGB'))
    img = gen_grayworld_image(img)
    Data_train[idx, :,:,:] = img
    #mask
    img2 = sc.imread(Data_list[idx]+'/' + name + mask_suffix + '/' +name + '_lesion.bmp')
    img2 = np.double(sc.imresize(img2, [height, width], interp='bilinear'))
    Label_train[idx, :,:] = img2    
         
print('Reading Ph2 finished')

# ################################################################ Make the train and test sets ########################################    
#  # We consider 140 samples for training, 20 samples for validation and 40 samples for testing


Train_img      = Data_train[0:140,:,:,:]
Validation_img = Data_train[140:160,:,:,:]
Test_img       = Data_train[160:200,:,:,:]

Train_mask      = Label_train[0:140,:,:]
Validation_mask = Label_train[140:160,:,:]
Test_mask       = Label_train[160:200,:,:]


np.save(Save_add+'data_train_gw', Train_img)
np.save(Save_add+'data_test_gw' , Test_img)
np.save(Save_add+'data_val_gw'  , Validation_img)

np.save(Save_add+'mask_train_gw', Train_mask)
np.save(Save_add+'mask_test_gw' , Test_mask)
np.save(Save_add+'mask_val_gw'  , Validation_mask)
