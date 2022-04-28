# coding:utf-8
# @Author     : HT
# @Time       : 2022/4/8 11:18
# @File       : compare_image_rgb.py
# @Software   : PyCharm
import os
import cv2
import numpy as np

def creat_dir(out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
image_path='G:\lunwen\image'
save_path=r'G:\mmdetection_miner_result/cat_image/'
creat_dir(save_path)
rgb_path = os.path.join(image_path, 'RGB')
rgb_path_list=os.listdir(rgb_path)
print(rgb_path_list)
depth_path = os.path.join(image_path, 'Depth')
rgb_depth_path = os.path.join(image_path, 'RGB-Depth')
rgb_path_list=[ 'RGBD_r_12_182.jpg','RGBD_r_13_455.jpg', 'RGBD_r_12_378.jpg', 'RGBD_r_13_563.jpg']
for i,rgbpath in enumerate(rgb_path_list):
    rgb_path_image = rgb_path+'/'+rgbpath
    print(rgb_path_image)
    rgb_path_image_data0=cv2.imread(rgb_path_image)
    depth_path_image = depth_path + '/' + rgbpath
    # print(rgb_path_image)
    depth_path_image_data0 = cv2.imread(depth_path_image)

    rgb_depth_path_image = rgb_depth_path + '/' + rgbpath
    # print(rgb_path_image)
    rgb_depth_path_image_data0 = cv2.imread(rgb_depth_path_image)
    image1 = np.vstack((rgb_path_image_data0, depth_path_image_data0))
    image2 = np.vstack((image1, rgb_depth_path_image_data0))
    if i==0:
        image3=image2
    else:
        image3 = np.hstack((image3, image2))

cv2.imshow('ht',image3)
cv2.waitKey(0)

save_path_image=save_path+'/'+'2.jpg'
cv2.imwrite(save_path_image,image3)




