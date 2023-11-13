# -*- coding:utf-8 -*-
'''
file    : _01_find_mask
author  : weihaoysgs@gmail.com
des     : $ adjust the H S V threshold
date    : 2021-03-28 15:06
IDE     : PyCharm
'''

import cv2
import numpy as np
import math


def nothing(x):
    pass


cap = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")
cv2.namedWindow("morphologyEX")
cv2.namedWindow('Canny')
# TODO : 添加关于霍夫圆变换的阈值部分，图像二值化的阈值部分。
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

cv2.createTrackbar("erode times", "morphologyEX", 0, 10, nothing)
cv2.createTrackbar("dilate times", "morphologyEX", 0, 10, nothing)

cv2.createTrackbar('gaussian size', 'Canny', 1, 20, nothing)
cv2.createTrackbar('canny H', 'Canny', 0, 255, nothing)
cv2.createTrackbar('canny L', 'Canny', 0, 255, nothing)

while True:

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    dilate_iteration_times = cv2.getTrackbarPos("erode times", "morphologyEX")
    erode_iteration_times = cv2.getTrackbarPos("dilate times", "morphologyEX")

    canny_low = cv2.getTrackbarPos('canny L', 'Canny')
    canny_high = cv2.getTrackbarPos('canny H', 'Canny')

    ret, frame = cap.read()
    gaussian_size = cv2.getTrackbarPos('gaussian size', 'Canny')
    if gaussian_size % 2 == 0:
        gaussian_size = gaussian_size + 1
    gaussian_frame = cv2.GaussianBlur(frame, (gaussian_size, gaussian_size), 0)
    hsv = cv2.cvtColor(gaussian_frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    # mask 是一个二值化图像
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # canny_mask = cv2.Canny(mask,canny_low,canny_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphologyEX_erode = cv2.erode(mask, kernel, iterations=erode_iteration_times)
    morphologyEX_dilate = cv2.dilate(morphologyEX_erode, kernel, iterations=dilate_iteration_times)

    result = cv2.bitwise_and(frame, frame, mask=mask)
    result = cv2.GaussianBlur(result, (gaussian_size, gaussian_size), 0)
    canny_result = cv2.Canny(result, canny_low, canny_high)
    # cv2.HoughCircles(canny_result,)

    cv2.imshow("frame", frame)
    cv2.imshow('gaussian_frame', gaussian_frame)
    cv2.imshow("mask", mask)
    cv2.imshow('canny_result', canny_result)
    cv2.imshow("result", result)
    cv2.imshow('morphologyEX_erode', morphologyEX_erode)
    cv2.imshow('morphologyEX_dilate', morphologyEX_dilate)
    key = cv2.waitKey(1)
    if key == 27:
        break

# cap.release()
cv2.destroyAllWindows()

lower_red = np.array([1, 64, 108])  # 红色阈值下界
higher_red = np.array([12, 158, 255])  # 红色阈值上界
img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask_red = cv2.inRange(img_hsv, lower_red, higher_red)
