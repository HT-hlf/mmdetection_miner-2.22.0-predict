# coding:utf-8
# @Author     : HT
# @Time       : 2022/4/13 22:26
# @File       : featuremap_geadcam++.py
# @Software   : PyCharm

import math
import torch

import cv2
import matplotlib.pyplot as plt
import numpy as np

from mmdet.apis import init_detector, inference_detector
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

from mmcv.parallel import collate, scatter
from mmcv.ops import RoIPool
from mmcv import Config, DictAction
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

config = 'work_dirs/yolo_v3/yolo_v3.py'
cfg = Config.fromfile(config)
checkpoint = 'work_dirs/yolo_v3/latest.pth'
device = 'cuda:0'
model = init_detector(config, checkpoint, device)

label_names = [
    'person bev', 'car bev', 'van bev', 'truck bev', 'bus bev',
    'person', 'car', 'aeroplane', 'bus', 'train', 'truck', 'boat',
    'bird', 'camouflage man'
]


dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False)


data = []
for i, t in enumerate(data_loader):
    tmp = {}
    tmp['img'] = t['img']
    tmp['img_metas'] = t['img_metas'][0].data[0]
    data.append(tmp)



image_path = "/home/cry/data4/Datasets/js-dataset/images/9999962_00000_d_0000088.jpg"
image = cv2.imread(image_path)
# scale = 600 / min(image.shape[:2])
image = cv2.resize(image,
                   dsize = (448, 448),
                   interpolation=cv2.INTER_AREA)
plt.figure(figsize=(7, 7))
plt.imshow(image[:, :, ::-1])
plt.show()


out = inference_detector(model, image)
res = image.copy()
for i, pred in enumerate(out):
    for *box, score in pred:
        if score < 0.4:
            break
        box = tuple(np.round(box).astype(int).tolist())
        print(i, label_names[i], box, score)
        cv2.rectangle(res, box[:2], box[2:], (0, 255, 0), 1)

plt.figure(figsize=(7, 7))
plt.imshow(res[:, :, ::-1])
plt.show()


class GradCAM_YOLOV3(object):
    """
    Grad CAM for Yolo V3 in mmdetection framework
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple
        :return:
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, data, index=0):
        """
        :param image: cv2 format, single image
        :param index: Which bounding box
        :return:
        """
        self.net.zero_grad()
        # Important
        feat = self.net.extract_feat(data['img'][0].cuda())
        res = model.bbox_head.simple_test(
            feat, data['img_metas'][0], rescale=True)

        score = res[0][0][index][4]

        score.backward()

        gradient = self.gradient.cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))[0]  # [C]

        feature = self.feature.cpu().data.numpy().squeeze()[0]  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        # Normalization
        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to 224*224
        box = res[0][0][index][:-1].cpu().detach().numpy().astype(np.int32)

        class_id = res[0][1][index].cpu().detach().numpy()
        return cam, box, class_id
def prepare_img(imgs):
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)
    print(datas)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    return data

def norm_image(image):
    """
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    # heatmap = np.float32(heatmap) / 255
    # heatmap = heatmap[..., ::-1]  # gbr to rgb

    # merge heatmap to original image
    cam = 0.5 * heatmap + 0.5 * image
    return norm_image(cam), heatmap

for name, m in model.named_modules():
    if isinstance(m, torch.nn.Conv2d):
        print(name,',',m)# backbone.conv_res_block5.res3.conv2.conv

# For example, you can choose 'backbone.conv_res_block4.conv.conv'
grad_cam = GradCAM_YOLOV3(model, 'backbone.conv_res_block4.conv.conv')

image_path = '/home/cry/data4/Datasets/js-dataset/images/0000008_02499_d_0000041.jpg'
image = cv2.imread(image_path)
data = prepare_img(image)

## First is the data, second is the index of the predicted bbox
mask, box, class_id = grad_cam(data,0)


mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
image_cam, heatmap = gen_cam(image, mask)