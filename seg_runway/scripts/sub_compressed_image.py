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
from paramerter import device

def callback(ros_data, args):
	
	np_arr = np.frombuffer(ros_data.data, np.uint8)
	img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
	cv2.imshow("racecar_image", img)
	cv2.waitKey(1)

def init_node():
	rospy.init_node('racecar_image_show', anonymous=True)
	compressed_image_subscriber = rospy.Subscriber("/image_view/image_raw/compressed",
												   CompressedImage, callback)
	rospy.spin()

if __name__ == '__main__':
	try:
		init_node()
	except rospy.ROSInterruptException:
		pass
