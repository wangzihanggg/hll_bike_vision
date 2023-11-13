import numpy as np
import cv2
import random as rng
import math

color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))

X_MID_VALUE = 128.0
ANGLE_MIN = 90.0


def post_process(image, debug):
	have_s_road = False
	angle = 0.0
	# this input is np.float32 type image, but the findContours function can only process np.uint8 image
	gray_image = (image * 255).astype(np.uint8)
	ret_val, threshold_image = cv2.threshold(gray_image, 230, 255, cv2.THRESH_BINARY)
	contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # only find the
	if len(contours) < 1 or len(contours) > 4:
		return have_s_road, angle

	drawing_image = np.zeros((threshold_image.shape[0], threshold_image.shape[1], 3), dtype=np.uint8)
	areas = [cv2.contourArea(contour) for contour in contours]
	biggest_contour = contours[np.argmax(areas)]
	#-----------------------------干净图像--------------------------------
	cv2.drawContours(drawing_image, [biggest_contour], -1, (255, 0, 0), cv2.FILLED)
	#-------------------------------------------------------------------
	biggest_contour = biggest_contour.reshape(-1, 2)

	center_points = biggest_contour[np.argwhere(biggest_contour[:, 1] > 128).ravel(), :]
	if len(center_points) < 2:
		return have_s_road, angle
	# center_points = biggest_contour[np.argwhere(center_points[:, 0] > 53 and center_points[:, 0] < 267).ravel(), :]
	# center_point = np.percentile(center_points, 50, axis=0, keepdims=True)[0]
	center_point = np.mean(center_points, axis=0)
	cv2.circle(drawing_image, (int(center_point[0]), int(center_point[1])), 1, (255, 255, 0), 1)

	angle = -ANGLE_MIN / X_MID_VALUE * center_point[0] + ANGLE_MIN #
	have_s_road = True
	print("center point x: ",center_point[0]," angle: ", angle)

	# output = cv2.fitLine(biggest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
	# k = output[1] / output[0]	have_s_road = False
	# this input is np.float32 type image, but the findContours function can only process np.uint8 image
	gray_image = (image * 255).astype(np.uint8)
	ret_val, threshold_image = cv2.threshold(gray_image, 230, 255, cv2.THRESH_BINARY)
	contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # only find the
	if len(contours) < 1 or len(contours) > 4:
		return have_s_road, angle

	if debug:
		cv2.imshow("draw", drawing_image)
		cv2.imshow("net_result", gray_image)
		cv2.imshow("threshold", threshold_image)
		cv2.waitKey(1)


if __name__ == '__main__':
	image = cv2.imread("./result/100.jpg")
	post_process(image, True)

