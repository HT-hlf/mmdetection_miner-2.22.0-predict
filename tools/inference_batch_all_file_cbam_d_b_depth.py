# coding:utf-8
# @Author     : HT
# @Time       : 2022/3/13 16:20
# @File       : inference.py
# @Software   : PyCharm

# encoding:utf-8
from mmdet.apis import init_detector, inference_detector
import os
import mmcv
import numpy as np
import cv2



if __name__ == '__main__':
    # config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    # checkpoint_file = 'G:\mmdetection_miner\check_file/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth'

    # config_file = 'configs/yolo/yolov3_d53_mstrain-416_273e_coco.py'
    # checkpoint_file = '../../mmdetection_miner_result\work_dirs\epoch_264.pth'

    # config_file = 'configs/yolo/yolov3_d53_mstrain-416_273e_coco_depth.py'
    # checkpoint_file = '../../mmdetection_miner_result\work_dirs_depth_simple\epoch_272.pth'

    # config_file = 'configs/yolo/yolov3_d53_mstrain-416_273e_coco_rgb_depth_diy_b.py'
    # checkpoint_file = '../../mmdetection_miner_result\work_dirs_rgb_depth_b_simple\epoch_272.pth'

    config_file = 'configs/yolo/yolov3_d53_mstrain-416_273e_coco_rgb_depth_attention_cbam_d_b.py'
    checkpoint_file = '../../mmdetection_miner_result\work_dirs_rgb_depth_attention_cbam_d_b_simple\epoch_162.pth'

    # 根据配置文件和 checkpoint 文件构建模型
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    # 测试单张图片并展示结果
    path=r'G:\mmdetection_miner\data\ht_cumt_rgbd\val2014'
    depth_path = r'G:\mmdetection_miner\data\ht_cumt_rgbd\depth_val'
    # path = r'G:\roadway_collect_dataset\recordData_process\RGBD_r_14\rgb'
    # depth_path = r'G:\roadway_collect_dataset\recordData_process\RGBD_r_14\depth'
    # path = r'G:\mmdetection_miner\mmdetection_miner-2.22.0-predict\image_test\rgb'
    # depth_path = r'G:\mmdetection_miner\mmdetection_miner-2.22.0-predict\image_test\depth'
    save_path=r'G:\mmdetection_miner\vritual_image_d'
    fps = 100
    imgInfo = (532, 1550)
    size = (imgInfo[1], imgInfo[0])  # 获取图片宽高度信息
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videoWrite = cv2.VideoWriter(r'G:\miner_detect_depth\sum_depth.mp4', fourcc, fps, size)  # 根据图片的大小，创建写入对象 （文件名，支持的编码器，5帧，视频大小（图片大小））

    for filename in os.listdir(path):
        img_rgb=path+'/'+filename
        img_depth = depth_path + '/' + filename
        print(filename)
        save_img = save_path + '/' + filename
        # img = r'..\data\ht_cumt_rgbd\test2014\RGBD_bk_5_237.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
        # result = inference_detector(model, img,img_prefix_miner='../data/ht_cumt_rgbd/train2014/',img_prefix_depth_miner='../data/ht_cumt_rgbd/depth_train/')
        result = inference_detector(model, filename, img_prefix_miner=path,
                                    img_prefix_depth_miner=depth_path)

        # 在一个新的窗口中将结果可视化
        img_rgb_depth_bbox=model.show_result_ht(img_rgb,img_depth, result,show=False,win_name='ht',
                    wait_time=1)
        # videoWrite.write(img_rgb_depth_bbox)  # 将图片写入所创建的视频对象
        # 或者将可视化结果保存为图片
        # model.show_result(img, result, out_file=save_img)

    # img='demo/demo.jpg'
    # result = inference_detector(model, img)