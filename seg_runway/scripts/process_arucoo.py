# -*- coding:utf-8 _*-

import numpy as np
import cv2
import cv2.aruco as aruco
import math

dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)  # 编码点的类型,与生成的时候对应
# 实际的 aruco 标记中点的坐标
EMap = np.loadtxt(
	'/home/highler/racecarVision_ws/src/visionmsg/detect_vehicle_line/EMap.txt')
dist_matrix = np.array(
	[-0.65846, 0.56120, -0.00665, -0.00634, 0.00000], dtype=np.float32)
K = np.array([[421.21833, 0, 316.54292],
			  [0, 419.28030,229.14683],
			  [0, 0, 1]], dtype=np.float32)

# 解算位姿的函数
def calculate_racecar_releative_camera_position(points_3d, points_2d, K):
	distortion = np.array([0, 0, 0, 0, 0.])
	ret, rvec_W2C, tvec_W2C = cv2.solvePnP(
		points_3d, points_2d, K, distortion)  # 解算位姿 -> 2d->3d
	RW2C = cv2.Rodrigues(rvec_W2C)[0]
	# RC2W = np.linalg.inv(RW2C)
	tC2W = -np.linalg.inv(RW2C).dot(tvec_W2C)
	New_tC2W = tC2W.flatten()  # 相机在世界坐标系下的坐标
	CamPosition = -RW2C.dot(New_tC2W)  # 世界坐标系在相机坐标系下的坐标
	return CamPosition


def deal_aruco_marker(Img, K, Show=False):
	CamPosition = []
	MarkerROI = None
	corners, IDs, _ = aruco.detectMarkers(Img, dict)
	dist_matrix = np.array([0, 0, 0, 0, 0.])
	rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
		corners, 0.1, K, dist_matrix)
	cv2.aruco.estimatePoseSingleMarkers()
	if len(corners) == 2:  # 如果检测点
		Point3D = np.empty((0, 3))
		Point2D = np.empty((0, 2))
		for i, Corner in enumerate(corners):
			ID = IDs.flatten()[i]
			if ID in [0, 1]:  # 防止错误检测ID
				Point2D = np.vstack((Point2D, Corner.reshape(-1, 2)))
				Point3D = np.vstack(
					(Point3D, np.hstack((EMap[ID, 3:].reshape(-1, 2), np.zeros((4, 1))))))
		CamPosition = calculate_racecar_releative_camera_position(
			Point3D, Point2D, K)
		if Show == True:
			aruco.drawDetectedMarkers(Img, corners, IDs)
			cv2.putText(Img, str(CamPosition), (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
		MarkerROI = np.hstack((np.min(Point2D, axis=0), np.max(
			Point2D, axis=0))).astype(np.int)  # xmin,ymin,xmax,ymax
	return CamPosition, MarkerROI


def get_aruco_distance(img, K, show=False):
	have_aruco = False
	aruco_distance = - 1.0
	corners, IDs, _ = aruco.detectMarkers(img, dict)
	rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
		corners, 0.1, K, dist_matrix)
	if len(corners) == 2:
		dis1 = math.sqrt(
			(math.pow(tvec[0][0][0], 2) + math.pow(tvec[0][0][1], 2) + math.pow(tvec[0][0][2], 2)))
		dis2 = math.sqrt(
			(math.pow(tvec[1][0][0], 2) + math.pow(tvec[1][0][1], 2) + math.pow(tvec[1][0][2], 2)))

		have_aruco = True
		aruco_distance = (dis1 + dis2) / 2.0
	
		if show == True:
			cv2.aruco.drawDetectedMarkers(img, corners, IDs)
			print("dis: ", (dis1 + dis2) / 2.0)
			cv2.putText(img, str('{:.2f}'.format(aruco_distance)), (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
			cv2.imshow("aruco_dis", img)
			cv2.waitKey(1)
	return have_aruco, aruco_distance

