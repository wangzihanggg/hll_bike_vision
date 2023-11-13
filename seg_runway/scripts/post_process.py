import cv2
import numpy as np
from collections import deque

area_threshold = 50
filter_list = deque([160]*30, maxlen=30)
prev_mid_point = None


def filter_avg(mid_point):
    global filter_list, prev_mid_point
    filter_list.append(mid_point)
    filtered_list = [val for val in filter_list if val is not None]
    if filtered_list:
        result_filtered = sum(filtered_list) / len(filtered_list)
        prev_mid_point = result_filtered
    else:
        result_filtered = prev_mid_point
    return result_filtered

def post_process(image, debug):
    global prev_mid_point
    mid_point = None  # 添加一个默认值为 None 的 mid_point

    gray_image = (image * 255).astype(np.uint8)
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
                if y == 100:
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

    prev_mid_point = mid_point
    mid_point = filter_avg(mid_point)
    cv2.circle(color_image, (int(mid_point), 100), 3, (255, 0, 0), 3)

    return mid_point, color_image

if __name__ == '__main__':
    image = cv2.imread("./result/0.jpg")
    mid_point, result_image = post_process(image, True)
    if mid_point is not None:
        print("Mid Point:", mid_point)
    cv2.imshow("Result Image", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()













