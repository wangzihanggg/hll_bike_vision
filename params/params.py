#!/home/wangzihanggg/miniconda3/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

unet_model_path = '/home/wangzihanggg/codespace/bike/bike_vision_ws/src/bike_vision/checkpoints/unet4.pth'

yolo_model_path = '/home/wangzihanggg/codespace/bike/bike_vision_ws/src/bike_vision/checkpoints/yolo.pth'
yolo_classes_path = '/home/wangzihanggg/codespace/bike/bike_vision_ws/src/bike_vision/checkpoints/bike_obstacle_classes.txt'
show_detect_result_img = True
show_astra_depth_img = False