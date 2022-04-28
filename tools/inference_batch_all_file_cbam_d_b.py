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

def show_result(self,
                img,
                result,
                score_thr=0.3,
                bbox_color=(72, 101, 241),
                text_color=(72, 101, 241),
                mask_color=None,
                thickness=2,
                font_size=13,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result (dict): The results.

        score_thr (float, optional): Minimum score of bboxes to be shown.
            Default: 0.3.
        bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: 'green'.
        text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: 'green'.
        mask_color (None or str or tuple(int) or :obj:`Color`):
           Color of masks. The tuple of color should be in BGR order.
           Default: None.
        thickness (int): Thickness of lines. Default: 2.
        font_size (int): Font size of texts. Default: 13.
        win_name (str): The window name. Default: ''.
        wait_time (float): Value of waitKey param.
            Default: 0.
        show (bool): Whether to show the image.
            Default: False.
        out_file (str or None): The filename to write the image.
            Default: None.

    Returns:
        img (Tensor): Only if not `show` or `out_file`.
    """
    img = mmcv.imread(img)
    img = img.copy()
    pan_results = result['pan_results']
    # keep objects ahead
    ids = np.unique(pan_results)[::-1]
    legal_indices = ids != self.num_classes  # for VOID label
    ids = ids[legal_indices]
    labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
    segms = (pan_results[None] == ids[:, None, None])

    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False
    # draw bounding boxes
    img = imshow_det_bboxes(
        img,
        segms=segms,
        labels=labels,
        class_names=self.CLASSES,
        bbox_color=bbox_color,
        text_color=text_color,
        mask_color=mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=show,
        wait_time=wait_time,
        out_file=out_file)

    if not (show or out_file):
        return img

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
    save_path=r'G:\mmdetection_miner\vritual_image_d'
    for filename in os.listdir(path):
        img=path+'/'+filename
        print(filename)
        save_img = save_path + '/' + filename
        # img = r'..\data\ht_cumt_rgbd\test2014\RGBD_bk_5_237.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
        # result = inference_detector(model, img,img_prefix_miner='../data/ht_cumt_rgbd/train2014/',img_prefix_depth_miner='../data/ht_cumt_rgbd/depth_train/')
        result = inference_detector(model, filename, img_prefix_miner='../data/ht_cumt_rgbd/val2014/',
                                    img_prefix_depth_miner='../data/ht_cumt_rgbd/depth_val/')

        # 在一个新的窗口中将结果可视化
        model.show_result(img, result,show=True,win_name='ht',
                    wait_time=1)
        # 或者将可视化结果保存为图片
        # model.show_result(img, result, out_file=save_img)

    # img='demo/demo.jpg'
    # result = inference_detector(model, img)