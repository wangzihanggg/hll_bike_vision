import cv2
import numpy as np
from collections import deque

orig_midpoint_y = 100
area_threshold = 50
filter_list = deque([160]*30, maxlen=30)
prev_mid_point = None

def filter_avg(mid_point):
    global filter_list
    if mid_point is not None:
        filter_list.append(mid_point)
        result_filted = sum(filter_list) / len(filter_list)
        return result_filted
    else:
        return None
def post_process(unet_image, color_image, depth_image, yolo_label, yolo_conf, yolo_boxes, debug=False):
    # yolo_process
    if yolo_label is None or yolo_conf is None or yolo_boxes is None:
        roadblock_distance = 0
    else:
        box = yolo_boxes
        top, left, bottom, right = box
        top = max(0, np.floor(top).astype('int32'))
        left = max(0, np.floor(left).astype('int32'))
        bottom = min(unet_image.shape[0], np.floor(bottom).astype('int32'))
        right = min(unet_image.shape[1], np.floor(right).astype('int32'))
        roadblock_y = int((top + bottom) / 2)
        roadblock_x = int((right + left) / 2)
        rect_roadblock = np.array([top, left, bottom, right], dtype=np.float32)
        roadblock_distance = int(depth_image[roadblock_y, roadblock_x] * 0.001)

    # unet_process
    global prev_mid_point
    mid_point = None
    gray_image = (unet_image * 255).astype(np.uint8)
    _, line_mask = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)
    color_image = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line1_x_sum = 0
    line1_count = 0
    line2_x_sum = 0
    line2_count = 0
    processing_line1 = True
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= area_threshold:
            for point in contour:
                x, y = point[0]
                if y == orig_midpoint_y:
                    if processing_line1:
                        line1_x_sum += x
                        line1_count += 1
                    else:
                        line2_x_sum += x
                        line2_count += 1
            processing_line1 = not processing_line1
    if line1_count > 0:
        line1_x_avg = line1_x_sum / line1_count
    else:
        line1_x_avg = None
    if line2_count > 0:
        line2_x_avg = line2_x_sum / line2_count
    else:
        line2_x_avg = None
    if line1_count > 0 and line2_count > 0:
        mid_point = (line1_x_avg + line2_x_avg) / 2
    if mid_point is None:
        mid_point = prev_mid_point
    if yolo_label is not None and roadblock_x <= int(mid_point):
        mid_point = roadblock_x + 70
    elif yolo_label is not None and roadblock_x > int(mid_point):
        mid_point = roadblock_x - 70
    mid_point = filter_avg(mid_point)
    prev_mid_point = mid_point
    servo_angle = int((mid_point - 160) * 0.86)

    # final process
    cv2.circle(color_image, (int(mid_point), orig_midpoint_y), 3, (255, 255, 0), 3)
    if yolo_label is not None and roadblock_y < orig_midpoint_y:
        cv2.circle(color_image, (roadblock_x, roadblock_y), 3, (0, 0, 255), 3)
    cv2.imshow('process_image', color_image)
    cv2.waitKey(1)
    return servo_angle, roadblock_distance

if __name__ == '__main__':
    image = cv2.imread("./result/0.jpg")
    mid_point, result_image = post_process(image, True)
    if mid_point is not None:
        print("Mid Point:", mid_point)
    cv2.imshow("Result Image", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()