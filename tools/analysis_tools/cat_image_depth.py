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
image_path='G:\lunwen\depth'
save_path=r'G:\mmdetection_miner_result/cat_image/'
creat_dir(save_path)
rgb_path_list=os.listdir(image_path)
# print(rgb_path_list)
# rgb_path_list=['RGBD_r_11_256.jpg', 'RGBD_r_11_362.jpg', 'RGBD_r_15_331.jpg', 'RGBD_r_15_79.jpg', 'RGBD_r_16_366.jpg', 'RGBD_r_1_101.jpg', 'RGBD_R_2_289.jpg', 'RGBD_r_6_44.jpg', 'RGBD_R_8_242.jpg']

# rgb_path_list=[
#                'RGBD_R_2_289.jpg', 'RGBD_r_6_44.jpg', 'RGBD_R_8_242.jpg']
for i,rgbpath in enumerate(rgb_path_list):

    rgb_path_image = image_path+'/'+rgbpath
    print(rgb_path_image)
    image2=cv2.imread(rgb_path_image)
    if i==0:
        image3=image2
    else:
        image3 = np.vstack((image3, image2))

    if i==2:
        save_path_image=save_path+'/'+'a3.jpg'
        cv2.imwrite(save_path_image,image3)

# cv2.imshow('ht',image3)
# cv2.waitKey(0)
#
# save_path_image = save_path + '/' + '9.jpg'
# cv2.imwrite(save_path_image, image3)






