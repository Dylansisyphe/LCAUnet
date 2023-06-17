'''
Descripttion: Created by QiSen Ma
Author: QiSen Ma
Date: 2023-01-16 01:49:39
LastEditTime: 2023-01-18 09:22:54
'''

import numpy as np
import scipy.io as sio
import scipy.misc as sc
import glob
from tqdm import tqdm


## configs
# image shape
height = 224
width  = 224
channels = 3
# folder path，修改为自己的文件路径
dataset_path = '/root/autodl-tmp/datasets/ISIC2018/'
save_path = '/root/autodl-tmp/datasets/ISIC2018/'

# read dataset
train_folder = 'ISIC2018_Task1-2_Training_Input'
label_folder = 'ISIC2018_Task1_Training_GroundTruth'
#glob.glob返回所有匹配的文件路径列表
Train_list = sorted(glob.glob(dataset_path+ train_folder+'/*.jpg'))
#共有2594个样本
data_train_2018    = np.zeros([2594, height, width, channels])
label_train_2018   = np.zeros([2594, height, width])
print('Begin reading ISIC 2018.')
for idx in tqdm(range(len(Train_list))):
    img = sc.imread(Train_list[idx])
    #是否能和cv2的库联动
    img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode = 'RGB'))
    data_train_2018[idx, :,:,:] = img

    b = Train_list[idx]   #b是路径
    a = b[0:len(dataset_path)]
    b = b[len(b)-16: len(b)-4] #b为图片编号ISIC_xxxxxxx
    label_path = (a+ label_folder + '/' + b +'_segmentation.png')
    img_label = sc.imread(label_path)
    img_label = np.double(sc.imresize(img_label, [height, width], interp='bilinear'))
    label_train_2018[idx, :,:] = img_label
print('Reading ISIC 2018 finished.')


print('Begin split the train and test sets')
# 1815 samples for training     0.7
# 259 samples for validation   0.1
# 520 samples for testing  0.2
train_img      = data_train_2018[0:1815,:,:,:]
validation_img = data_train_2018[1815:1815+259,:,:,:]
test_img       = data_train_2018[1815+259:2594,:,:,:]
np.save(save_path + 'data_train', train_img)
np.save(save_path + 'data_test' , test_img)
np.save(save_path + 'data_val'  , validation_img)

train_mask      = label_train_2018[0:1815,:,:]
validation_mask = label_train_2018[1815:1815+259,:,:]
test_mask       = label_train_2018[1815+259:2594,:,:]
np.save(save_path + 'mask_train', train_mask)
np.save(save_path + 'mask_test' , test_mask)
np.save(save_path + 'mask_val'  , validation_mask)




