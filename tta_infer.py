#Author: chenfan_qu@qcf-568
import os
import cv2
import json
import mmcv
import torch
import pickle
import argparse
import numpy as np
from tqdm miport tqdm
from mmengine import ConfigDict
from mmengine.config import Config
from mmdet.utils import register_all_modules
from rcnn_apis import init_detector, inference_detector
from ensemble_boxes import weighted_boxes_fusion # pip install ensemble_boxes

parser = argparse.ArgumentParser(description='Train a segmentor')
parser.add_argument('--cfg', type=str) # 推理配置文件, 如bisai.py
parser.add_argument('--pth', type=str) # 训好的模型pth
parser.add_argument('--sz', type=str, default='800,1333') # 基本尺度, 这个不用动
args = parser.parse_args()

config_file = args.cfg
checkpoint_file = args.pth

def unnorm_box(box, w, h):
    if len(box)==0:
        return box
    else:
        box[:,0] = box[:,0]*w
        box[:,2] = box[:,2]*w
        box[:,1] = box[:,1]*h
        box[:,3] = box[:,3]*h
        return box

def norm_box(box, w, h):
    if len(box)==0:
        return box
    else:
        box[:,0] = box[:,0]/w
        box[:,1] = box[:,1]/h
        box[:,2] = box[:,2]/w
        box[:,3] = box[:,3]/h
        return box

register_all_modules()
# build the model from a config file and a checkpoint file
config_file = Config.fromfile(config_file)
config_file.model.test_cfg['rpn']['nms_pre']=5000
config_file.model.test_cfg['rpn']['max_per_img']=5000
config_file.model.test_cfg['rcnn']['score_thres']=0.01
config_file.model.test_cfg['rcnn']['max_per_img']=10
config_file.model.pretrained = args.pth
config_file.model = ConfigDict(**config_file.tta_model, module=config_file.model)

test_data_cfg = config_file.test_dataloader.dataset
while 'dataset' in test_data_cfg:
    test_data_cfg = test_data_cfg['dataset']
if 'batch_shapes_cfg' in test_data_cfg:
    test_data_cfg.batch_shapes_cfg = None
test_data_cfg.pipeline = config_file.tta_pipeline
s1,s2 = args.sz.split(',')
assert (test_data_cfg.pipeline[1]['transforms'][0][0]['type']=='Resize')
test_data_cfg.pipeline[1]['transforms'][0][0]['scale']=(int(s1), int(s2))

model = init_detector(config_file, checkpoint_file, device='cuda:0',cfg_options={})

for i,file_path in enumerate(tqdm(os.listdir(test_img_dir))): # test_img_dir改成本地测试图片路径
    img = cv2.imread(file_path)
    h,w = img.shape[:2]
    result = inference_detector(model, img)
    v = result[0]
    boxes = [norm_box(vv[0], w, h).tolist() for vv in v]
    scores = [vv[1].tolist() for vv in v]
    labels = [vv[2].tolist() for vv in v]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=[1 for x in range(len(boxes))], iou_thr=0.25, skip_box_thr=0.0001)
    # 自定义结果汇总保存方法
