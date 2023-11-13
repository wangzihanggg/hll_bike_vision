#!/home/wangzihanggg/miniconda3/envs/pytorch/bin/python
# -*- coding: utf-8 -*-
run_dir = "/home/wangzihanggg/codespace/bike/bike_vision_ws/src"

import sys
sys.path.append("{}/bike_vision/seg_runway/scripts".format(run_dir))
sys.path.append("{}/bike_vision/detect_roadblock/scripts".format(run_dir))

from pydoc import importfile
import time
import cv2
import numpy as np
import os
import torch
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
import rospy
from cv2 import getTickCount
from collections import deque
from termcolor import colored

from seg_runway.scripts.detect_one_image import detect_one_image
from seg_runway.scripts.net.unet_bigger import UNetBig
from detect_roadblock.scripts.bike_yolo import YOLO
from params import device, unet_model_path, yolo_model_path, yolo_classes_path
from post_process import post_process
from bike_vision.msg import vision_msg

class BikeVision():
	def __init__(self):
		self.debug = False
		self.device = device
		self.unet_model = self.unet_init(unet_model_path)
		self.yolo_model = self.yolo_init(yolo_model_path, yolo_classes_path)
		self.init_subscriber()
		self.init_publisher()

	def unet_init(self, weight_path):
		if os.path.exists(weight_path):
			net = UNetBig(in_channels=3, out_channels=1, init_features=8, WithActivateLast=False).to(device)
			net.load_state_dict(torch.load(weight_path, map_location=device))
			print(colored('Unet Init Success!', 'green'))
			return net
		else:
			print(colored('Unet Init Failed!', 'red'))
			return
		
	def yolo_init(self, weight_path, classes_path):
		if os.path.exists(weight_path) & os.path.exists(classes_path):
			net = YOLO(weight_path, classes_path)
			print(colored('YOLO Init Success!', 'green'))
			return net
		else:
			print(colored('Unet Init Failed!', 'red'))
			return
		
	def astra_color_to_cv2(self, img_msg):
		if img_msg.encoding != "rgb8":
			rospy.logerr(
				"This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're "
				"actually trying to implement a new camera")
		dtype = np.dtype("uint8")
		dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
		image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3),
									dtype=dtype, buffer=img_msg.data)
		if img_msg.is_bigendian == (sys.byteorder == 'little'):
			image_opencv = image_opencv.byteswap().newbyteorder()
		return image_opencv
	
	def astra_depth_to_cv2(self, depth_img_msg):
		if depth_img_msg.encoding != "16UC1":
			rospy.logerr("This depth image is not encoding with U16C1")
		dtype = np.dtype("uint16")
		dtype = dtype.newbyteorder('>' if depth_img_msg.is_bigendian else '<')
		depth_cv_img = np.ndarray(shape=(depth_img_msg.height, depth_img_msg.width, 1),
									dtype=dtype, buffer=depth_img_msg.data)
		if depth_img_msg.is_bigendian == (sys.byteorder == 'little'):
			depth_cv_img = depth_cv_img.byteswap().newbyteorder()
		return depth_cv_img

	def publish_detect_result_img(self, cv_result_img):
		msg = CompressedImage()
		msg.header.stamp = rospy.Time.now()
		msg.format = "jpeg"
		msg.data = np.array(cv2.imencode('.jpg', cv_result_img)[1]).tostring()
		# Publish new image
		self.pub_result_img.publish(msg)

	def publish_detect_result_to_ros(self, servo_angle, roadblock_distance):
		vision_status = vision_msg()
		vision_status.header = rospy.Header()
		vision_status.servo_angle = servo_angle
		vision_status.roadblock_distance = roadblock_distance
		self.pub_vision_status.publish(vision_status)

	def sub_astra_color_depth_callback(self, astra_color_img):
		cv_color_img = self.astra_color_to_cv2(astra_color_img)
		small_image, unet_image = detect_one_image(cv_color_img, self.unet_model, self.device)
		small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
		yolo_label, yolo_conf, yolo_boxes = None, None, None
		yolo_label, yolo_conf, yolo_boxes = self.yolo_model.detect_image(small_image)
		servo_angle, roadblock_distance = post_process(unet_image, small_image, self.cv_depth_img, yolo_label, yolo_conf, yolo_boxes, self.debug)
		self.publish_detect_result_to_ros(servo_angle, roadblock_distance)
		cv2.waitKey(1)
		
		
	def sub_depth_image_callback(self, astra_depth_img):
		self.cv_depth_img = self.astra_depth_to_cv2(astra_depth_img)
	
	def init_subscriber(self):
		sub_color_image = rospy.Subscriber("/camera/color/image_raw", Image, self.sub_astra_color_depth_callback)
		sub_depth_image = rospy.Subscriber("/camera/depth/image_raw", Image, self.sub_depth_image_callback)

	def init_publisher(self):
		self.pub_vision_status = rospy.Publisher("bike_vision_status", vision_msg, queue_size=10)


if __name__ == '__main__':
	try:
		rospy.init_node("bike_vision_node")
		bike_vision = BikeVision()
		rospy.spin()
	except rospy.ROSInterruptException:
		pass