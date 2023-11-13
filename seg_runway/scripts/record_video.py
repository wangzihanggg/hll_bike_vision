#!/home/wangzihanggg/miniconda3/envs/pytorch/bin/python
# -*-. coding: utf-8 -*-
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

import cv2
import numpy as np
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time
from sensor_msgs.msg import Image
import ros_numpy
import time

# Define global variables for video writer
video_name = 'output_video_{}.mp4'.format(time.time())
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec as needed
out = cv2.VideoWriter(video_name, fourcc, 20.0, (640, 480))  # Adjust resolution and frame rate

def callback(ros_data):
    global out
    image_np = ros_numpy.numpify(ros_data)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    cv2.imshow('cv_img', image_np)
    cv2.waitKey(2)
    out.write(image_np)
    print('saved one image at {}'.format(time.time()))

def main():
    rospy.init_node('camera_sub_node', anonymous=True)  # 定义节点
    compressed_image_subscriber = rospy.Subscriber("/camera/color/image_raw",
                                                   Image, callback)
    rospy.spin()

if __name__ == "__main__":
    main()
