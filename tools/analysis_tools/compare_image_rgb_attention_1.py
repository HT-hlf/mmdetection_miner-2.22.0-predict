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
sum_path_list=[r'G:\mmdetection_miner_result\work_dirs_rgb_depth_diy_d_simple\val\drise',r'G:\mmdetection_miner_result\work_dir_rgb_depth_attention_se_d_simple\val\drise',r'G:\mmdetection_miner_result\work_dirs_rgb_depth_attention_cbam_d_b_simple\val\drise']

# sum_path_list=[r'G:\mmdetection_miner_result\work_dirs_rgb_simple',r'G:\mmdetection_miner_result\work_dirs_depth_simple','G:\mmdetection_miner_result\work_dir_rgb_depth_attention_se_b_simple','G:\mmdetection_miner_result\work_dir_rgb_depth_attention_se_just_four_simple']
# sum_path_list=[r'G:\mmdetection_miner_result\work_dirs_rgb_simple',r'G:\mmdetection_miner_result\work_dirs_depth_simple','work_dir_rgb_depth_attention_se_b_simple','work_dir_rgb_depth_attention_se_just_four_simple']
save_path=r'G:\mmdetection_miner_result/compare_image_attention_drise'
creat_dir(save_path)
rgb_path = os.path.join(sum_path_list[0],'rgb')
rgb_path_list=os.listdir(rgb_path)
depth_path = os.path.join(sum_path_list[0],'depth')
for rgbpath in rgb_path_list:
    rgb_path_image = rgb_path+'/'+rgbpath
    # print(rgb_path_image)
    rgb_path_image_data0=cv2.imread(rgb_path_image)
    depth_path_image = depth_path + '/' + rgbpath
    # print(rgb_path_image)
    depth_path_image_data0 = cv2.imread(depth_path_image)
    img_shape=rgb_path_image_data0.shape
    # print('img_shape',img_shape)
    if depth_path_image_data0 is None:
        # print('ht')
        depth_path_image_data0 = np.uint8(np.ones(img_shape) * 255)
    for sum_path in sum_path_list[1:]:
        sum_path_ele_rgb = os.path.join(sum_path, 'rgb')
        sum_path_ele_rgb_image=sum_path_ele_rgb+'/'+rgbpath
        # print(sum_path_ele_rgb_image)
        rgb_path_image_data1=cv2.imread(sum_path_ele_rgb_image)
        if rgb_path_image_data1 is None:
            rgb_path_image_data1=np.uint8(np.ones(img_shape)*255)
        # print(rgb_path_image_data1.shape)

        sum_path_ele_depth = os.path.join(sum_path, 'depth')
        sum_path_ele_depth_image = sum_path_ele_depth + '/' + rgbpath
        depth_path_image_data1 = cv2.imread(sum_path_ele_depth_image)
        if depth_path_image_data1 is None:
            depth_path_image_data1=np.uint8(np.ones(img_shape)*255)
        # print('depth_path_image_data1', depth_path_image_data1.shape)
        # print(depth_path_image_data1)
        # print(rgb_path_image_data1.shape)
        rgb_path_image_data0 = np.hstack((rgb_path_image_data0, rgb_path_image_data1))
        depth_path_image_data0 = np.hstack((depth_path_image_data0, depth_path_image_data1))
        # cv2.imshow('ht1', rgb_path_image_data0)
        # cv2.imshow('ht2', depth_path_image_data0)
    rgb_depth_path_image_data0 = np.vstack((rgb_path_image_data0, depth_path_image_data0))
    # cv2.imshow('ht',rgb_depth_path_image_data0)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    save_path_image=save_path+'/'+rgbpath
    cv2.imwrite(save_path_image,rgb_depth_path_image_data0)




