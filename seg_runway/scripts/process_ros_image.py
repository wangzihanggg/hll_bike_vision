#!/home/highler/.conda/envs/YOLOv5_38/bin/python
# -*- coding: utf-8 -*-

import sys
from pydoc import importfile
import time
import cv2
import numpy as np
import os
import torch
from sensor_msgs.msg import CompressedImage
import rospy
from std_msgs.msg import String
from detect_one_image import detect_one_image
from net.unet_bigger import UNetBig
from paramerter import device
from visionmsg.msg import trafficlight 
from visionmsg.msg import sroadline
from visionmsg.msg import arucostatus
from visionmsg.msg import parkingstatus
from process_arucoo import get_aruco_distance , K
from detect_traffic_light.predictReceiveImage import predict as predict_trafficLight
from detect_parking_text.predictReceiveImage import predict as predict_parkingText
from cv2 import getTickCount, getTickFrequency
from collections import deque


i = 0

def filter(result):
	filter = deque(maxlen=15)
	filter.append(result)
	result_filted = max(filter,key=filter.count)
	return result_filted
	

def process_traffic_light(image, debug = True):
	result, ret_img = predict_trafficLight(image)
	print('traffic_light_status: {}'.format(result))
	if debug and result!=-1:
		img = cv2.cvtColor(np.array(ret_img), cv2.COLOR_RGB2BGR)
		cv2.imshow("traffic", img)
		cv2.waitKey(1)
	if result!=-1:
		return result
	else:
		return -1


def process_parking_text(image, debug = True):
	result, ret_img = predict_parkingText(image)
	if debug and result!=-1:
		img = cv2.cvtColor(np.array(ret_img), cv2.COLOR_RGB2BGR)
		cv2.imshow("parking", img)
		cv2.waitKey(1)

	if result == 10 or result == 30:
		result = 1
	elif result == 20 or result == 40:
		result = 2
	else:
		result = -1
	
	print('parking_status: {}'.format(result))

	return result


def init_model(weight_path):
	net = UNetBig(in_channels=3, out_channels=1, init_features=8, WithActivateLast=False).to(device)
	if os.path.exists(weight_path):
		net.load_state_dict(torch.load(weight_path, map_location=device))
		print('successfully load weight')
		return net
	else:
		print('not correct loading weight')
		return


def callback(ros_data, args):
	global i
	net = args[0]
	pub_scorner_angle = args[1]
	# pub_traffic_light = args[2]
	pub_aruco_status = args[2]
	pub_parking_status = args[3]
	loop_start = getTickCount()
	
	np_arr = np.frombuffer(ros_data.data, np.uint8)
	img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

	pressedKey = cv2.waitKey(1) & 0xFF
	if pressedKey == ord('s'):  # 按’s‘ 保存图片C
		print('saved image to {}'.format("/home/highler/racecarVision_ws/src/visionmsg/detect_vehicle_line/racecar_save_imgs_2023/"+ str(i) + ".jpg"))
		cv2.imwrite("/home/highler/racecarVision_ws/src/visionmsg/detect_vehicle_line/racecar_save_imgs_2023/"+ str(i) + ".jpg", img)  # 保存路径
		i = i + 1


	#处理红绿灯
	# have_aruco, aruco_dis = get_aruco_distance(img, K, False)
	aruco_status_msg = arucostatus()
	# aruco_status_msg.aruco_distance = aruco_dis
	# aruco_status_msg.have_aruco = have_aruco  
	aruco_status_msg.aruco_distance = 0
	aruco_status_msg.have_aruco = 0
	aruco_result = process_traffic_light(img, False)
	# aruco_result = filter(aruco_result)
	aruco_status_msg.trafficlight_color = aruco_result
	pub_aruco_status.publish(aruco_status_msg)

	#处理文字识别
	parking_status_msg = parkingstatus()
	parking_status_msg.parkingStatus = process_parking_text(img, False)
	pub_parking_status.publish(parking_status_msg)
	

	# 处理 S 弯道
	# img[:270,:,:] = 0
	have_s_road, angle = detect_one_image(img, net, device, False)
	if angle < 3 and angle >= -10:
		angle = angle - 10
	elif angle <= -10:
		angle = angle * 1.25
	else:
		angle = angle * 1  # 0.8->0.95

	sroad_status_msg = sroadline()
	sroad_status_msg.trafficstatus = have_s_road
	sroad_status_msg.angle = angle
	pub_scorner_angle.publish(sroad_status_msg)
	
	loop_time = cv2.getTickCount() - loop_start

	total_time = loop_time/(cv2.getTickFrequency()) # 使用getTickFrequency()更加准确
	fps = int(1 / total_time) # 帧率取整
	img = cv2.putText(img, "FPS:" + str(fps), (50, 50), 
					  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
	cv2.imshow("racecar_image", img)
	cv2.waitKey(1)

def init_node():
	rospy.init_node('racecar_vision', anonymous=True)
	pub_scorner_angle = rospy.Publisher('sroadline_status', sroadline, queue_size=1)
	# pub_traffic_light = rospy.Publisher('traffic_light', String, queue_size=1) # 无用
	pub_aruco_status = rospy.Publisher('aruco_status', arucostatus, queue_size=1)
	pub_parking_status = rospy.Publisher('parking_status', parkingstatus, queue_size=1)
	net = init_model(
		'/home/highler/racecarVision_ws/src/visionmsg/detect_vehicle_line/pth/2023_epoch_275_loss_0.0007359060691669583.pth')
	compressed_image_subscriber = rospy.Subscriber("/image_view/image_raw/compressed",
												   CompressedImage, callback,
												   (net, pub_scorner_angle, pub_aruco_status, pub_parking_status))
	rospy.spin()


if __name__ == '__main__':
	try:
		init_node()
	except rospy.ROSInterruptException:
		pass
