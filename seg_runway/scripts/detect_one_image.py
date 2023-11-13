#!/home/wangzihanggg/miniconda3/envs/pytorch/bin/python
# -*- coding:utf-8 _*-
import torch
from utils.resize_image_keep_scale import resize_image_fixed
from utils.transform import train_transform

def detect_one_image(frame, net, device):
	with torch.no_grad():
		small_frame = resize_image_fixed(frame, size=(320, 240))
		frame = train_transform(small_frame).to(device)
		frame = torch.unsqueeze(frame, dim=0)
		out_frame = net(frame)
		out_frame = out_frame.cpu().numpy().squeeze(0).transpose((1, 2, 0))
		return small_frame, out_frame
