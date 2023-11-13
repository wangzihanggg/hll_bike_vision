# -*- coding:utf-8 _*-
import cv2
import torch
from PIL import Image
import time
from net.unet_bigger import UNetBig
import os
from utils.resize_image_keep_scale import resize_image_fixed
import numpy as np
from torchvision.utils import save_image
from utils.transform import train_transform
from post_process import post_process


video_path = "/home/wangzihanggg/codespace/bike/bike_vision_ws/src/bike_vision/seg_runway/scripts/output_video_1699871809.7931747.mp4"
weights_path = '/home/wangzihanggg/codespace/bike/bike_vision_ws/src/bike_vision/seg_runway/pth/unet.pth'

fps = 0.0


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using device ", device)
    net = UNetBig(in_channels=3, out_channels=1, init_features=8, WithActivateLast=False).to(device)
    if os.path.exists(weights_path):
        net.load_state_dict(torch.load(weights_path, map_location=device))
        print('successfully loaded weight')
    else:
        print('no loading')

    video_capture = cv2.VideoCapture(video_path)
    ret, frame = video_capture.read()
    if not ret:
        raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）")
    i = 0

    with torch.no_grad():
        while True:
            start_time = time.time()
            ref, frame = video_capture.read()
            if not ref:
                break
            cv2.imshow("camera", frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = resize_image_fixed(frame, size=(320, 240))

            frame = train_transform(frame).to(device)
            frame = torch.unsqueeze(frame, dim=0)

            out_frame = net(frame)

            # if i % 20 == 0:
            #     save_image(out_frame, f"./result/{i}.jpg")
            out_frame = out_frame.cpu().numpy().squeeze(0).transpose((1, 2, 0))
            mid_point, mid_point_image = post_process(out_frame, debug=False)
            fps = (fps + (1. / (time.time() - start_time))) / 2
            if mid_point:
                print('\033[92m' + 'fps: {:.2f} ; mid_point: {:.2f}'.format(fps, mid_point) + '\033[0m')
            cv2.imshow('seg', out_frame)
            cv2.imshow('mid_point', mid_point_image)

            c = cv2.waitKey(1) & 0xff
            if c == 27:
                video_capture.release()
                print("Video Detection Done!")
                break
            i = i + 1
