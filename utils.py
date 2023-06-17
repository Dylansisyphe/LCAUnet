'''
Descripttion: Created by QiSen Ma
Author: QiSen Ma
Date: 2023-01-15 22:59:56
LastEditTime: 2023-02-04 09:55:28
'''
import cv2
import os
import matplotlib.pyplot as plt

def Edge_Extract(image_path):
    #0表示以灰度图的形式读取图片
    #读取后的图片以多维数组的形式保存图片信息，HWC为索引顺序
    img = cv2.imread(image_path,0)

    # print(image_name)
    edge = cv2.Canny(img,30,100)
    cv2.imwrite(image_path+"_edge.png", edge)


def draw_contrast(img,mask):
    if img.shape[0] == 3:
        img = img.permute(1,2,0)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("img")
    plt.imshow(img) #应该是默认的RGB显示方式
    print('img.shape',img.shape)

    plt.subplot(1, 2, 2)
    plt.title("mask")
    plt.imshow(mask, cmap="gray") #设置为灰度图的显示方式
    print('mask.shape',mask.shape)
    plt.show()


def test_dataset(dataset):
    print(len(dataset))

    sample = dataset[int(len(dataset)/6)]
    print('case_name:',sample['case_name'])
    sample_img = sample['image']
    sample_mask = sample['mask']
    draw_contrast(sample_img,sample_mask)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count