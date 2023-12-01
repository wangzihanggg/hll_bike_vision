import cv2
import numpy as np
from collections import deque

area_threshold = 50
filter_list = deque([160]*20, maxlen=20)
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

def perspective_transform(image):
    height, width = image.shape[:2]
    src_points = np.float32([[20, 175], [100, 20], [190, 20], [300, 175]])
    dst_points = np.float32([[80, 240], [20, 0], [300, 0], [240, 240]])
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    birdseye_image = cv2.warpPerspective(image, perspective_matrix, (width, height), flags=cv2.INTER_LINEAR)
    return birdseye_image, perspective_matrix

def fit_polynomial(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 9
    window_height = np.int(binary_warped.shape[0] // nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 50
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                    nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    return left_fit, right_fit, left_lane_inds, right_lane_inds

# def calculate_curvature_and_offset(binary_image, three_ch_image, left_fit, right_fit, perspective_matrix, express_y, yolo_label, roadblock_y, roadblock_x):
#     img_size = (binary_image.shape[1], binary_image.shape[0])
#     dist_from_center = 0.0
#     if right_fit is not None and left_fit is not None:
#         camera_pos = img_size[0] / 2
#         if yolo_label is not None:
#             birdeye_cam = cv2.perspectiveTransform(np.array([[[camera_pos, 0]]], dtype=np.float32), perspective_matrix)[0][0][0]
#             roadblock_coords = np.array([[roadblock_x, roadblock_y]], dtype=np.float32)
#             birdseye_coords = cv2.perspectiveTransform(roadblock_coords.reshape(-1, 1, 2), np.linalg.inv(perspective_matrix))
#             birdseye_roadblock_x = int(birdseye_coords[0][0][0])
#             birdseye_roadblock_y = int(birdseye_coords[0][0][1])
#             print("cam: {}, roadblock_x:{}, roadblock_y: {}".format(camera_pos, roadblock_x, roadblock_y))
#             print("birdeye_cam: {}, birdeye_roadblock_x:{}, birdeye_roadblock_y: {}".format(birdeye_cam, birdseye_roadblock_x, birdseye_roadblock_y))
#             cv2.circle(three_ch_image, (birdseye_roadblock_x, birdseye_roadblock_y), 3, (0, 255, 0), 3)
#             if birdseye_roadblock_x >= birdeye_cam:
#                 left_lane_pix = np.polyval(left_fit, birdseye_roadblock_y)
#                 center_of_lane_pix = (left_lane_pix + birdseye_roadblock_x) / 2
#                 birdseye_point = np.array([[center_of_lane_pix, birdseye_roadblock_y]], dtype=np.float32)
#             else:
#                 right_lane_pix = np.polyval(right_fit, birdseye_roadblock_y)
#                 center_of_lane_pix = (right_lane_pix + birdseye_roadblock_x) / 2
#                 birdseye_point = np.array([[center_of_lane_pix, birdseye_roadblock_y]], dtype=np.float32)
#             cv2.circle(three_ch_image, (int(center_of_lane_pix), int(birdseye_roadblock_y)), 3, (255, 0, 0), 3)
#         else:
#             birdseye_y = cv2.perspectiveTransform(np.array([[[0, express_y]]], dtype=np.float32), perspective_matrix)[0][0][1]
#             left_lane_pix = np.polyval(left_fit, birdseye_y)
#             right_lane_pix = np.polyval(right_fit, birdseye_y)
#             center_of_lane_pix = (left_lane_pix + right_lane_pix) / 2
#             birdseye_point = np.array([[center_of_lane_pix, birdseye_y]], dtype=np.float32)
#             cv2.circle(three_ch_image, (int(center_of_lane_pix), int(birdseye_y)), 3, (255, 0, 0), 3)
#         original_point = cv2.perspectiveTransform(birdseye_point.reshape(-1, 1, 2), np.linalg.inv(perspective_matrix))
#         original_center_x = original_point[0][0][0]
#         original_center_y = original_point[0][0][1]
#         # print("Birdseye View - Lane Center at y=100:", center_of_lane_pix)
#         # print("Original Image - Lane Center at y=100:", original_center_x, original_center_y)
#         # dist_from_center = camera_pos - original_center_x
#     return int(original_center_x)

def calculate_curvature_and_offset(binary_image, left_fit, right_fit, perspective_matrix, express_y):
    img_size = (binary_image.shape[1], binary_image.shape[0])
    dist_from_center = 0.0
    if right_fit is not None and left_fit is not None:
        camera_pos = img_size[0] / 2
        birdseye_y = cv2.perspectiveTransform(np.array([[[0, express_y]]], dtype=np.float32), perspective_matrix)[0][0][1]
        left_lane_pix = np.polyval(left_fit, birdseye_y)
        right_lane_pix = np.polyval(right_fit, birdseye_y)
        center_of_lane_pix = (left_lane_pix + right_lane_pix) / 2
        # print("Birdseye View - Lane Center at y=100:", center_of_lane_pix)
        birdseye_point = np.array([[center_of_lane_pix, birdseye_y]], dtype=np.float32)
        original_point = cv2.perspectiveTransform(birdseye_point.reshape(-1, 1, 2), np.linalg.inv(perspective_matrix))
        original_center_x = original_point[0][0][0]
        original_center_y = original_point[0][0][1]
        # print("Original Image - Lane Center at y=100:", original_center_x, original_center_y)
        # dist_from_center = camera_pos - original_center_x
    return int(original_center_x)

def draw_lane_lines(original_image, binary_warped, left_fit, right_fit, perspective_matrix, mid_point):
    img_size = (original_image.shape[1], original_image.shape[0])
    camera_pos = img_size[0] / 2
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    pts = cv2.perspectiveTransform(pts, np.linalg.inv(perspective_matrix))
    cv2.polylines(original_image, np.int32([pts]), isClosed=False, color=(255, 0, 0), thickness=3)
    cv2.circle(original_image, (int(mid_point), 100), 3, (0, 255, 0), -1)
    cv2.circle(original_image, (int(camera_pos), 100), 3, (0, 0, 255), -1)
    cv2.line(original_image, (int(mid_point), 100), (int(camera_pos), 100), (255, 255, 0), thickness=2)
    return original_image

def detect_yellow_tape(image, yellow_tape_detected_first):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define lower and upper bounds for yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    # Threshold the image to extract yellow pixels
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    # Check if there are enough yellow pixels indicating yellow tape
    yellow_pixel_count = cv2.countNonZero(yellow_mask)
    yellow_threshold = 1000  # Adjust this threshold based on your needs
    cv2.imshow('hsv', hsv_image)
    if yellow_pixel_count > yellow_threshold:
        if not yellow_tape_detected_first:
            # 第一次检测到黄色胶带
            yellow_tape_detected_first = True
            stop_bike = False
            return stop_bike, yellow_tape_detected_first, yellow_mask
        else:
            # 第二次检测到黄色胶带
            stop_bike = True
            return stop_bike, yellow_tape_detected_first, yellow_mask
    else:
        stop_bike = False
        return stop_bike, yellow_tape_detected_first, yellow_mask


def post_process(unet_img, color_img, depth_img, yolo_label, yolo_conf, yolo_boxes, debug):
    # yolo_process
    if yolo_label is None or yolo_conf is None or yolo_boxes is None:
        roadblock_distance, roadblock_y, roadblock_x = 0, None, None
    else:
        box = yolo_boxes
        top, left, bottom, right = box
        top = max(0, np.floor(top).astype('int32'))
        left = max(0, np.floor(left).astype('int32'))
        bottom = min(unet_img.shape[0], np.floor(bottom).astype('int32'))
        right = min(unet_img.shape[1], np.floor(right).astype('int32'))
        roadblock_y = int((top + bottom) / 2)
        roadblock_x = int((right + left) / 2)
        rect_roadblock = np.array([top, left, bottom, right], dtype=np.float32)
        roadblock_distance = int(depth_img[roadblock_y, roadblock_x] * 0.001)

    # unet_process
    global prev_mid_point
    mid_point = None
    express_y = 100
    gray_image = (unet_img * 255).astype(np.uint8)
    _, line_mask = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)
    mask_three_ch_image = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2BGR)
    birdseye_image, perspective_matrix = perspective_transform(line_mask)
    birdseye_three_ch_image = cv2.cvtColor(birdseye_image, cv2.COLOR_GRAY2BGR)
    left_line_fit, right_line_fit, left_lane_inds, right_lane_inds = fit_polynomial(birdseye_image)
    # mid_point = calculate_curvature_and_offset(birdseye_image, birdseye_three_ch_image, left_line_fit, right_line_fit, perspective_matrix, express_y, yolo_label, roadblock_y, roadblock_x)
    mid_point = calculate_curvature_and_offset(birdseye_image, left_line_fit, right_line_fit, perspective_matrix, express_y)
    if mid_point is None:
        mid_point = prev_mid_point
    prev_mid_point = mid_point

    # final process
    if yolo_label is not None and roadblock_x <= int(mid_point):
        mid_point = roadblock_x + 30
    elif yolo_label is not None and roadblock_x > int(mid_point):
        mid_point = roadblock_x - 30
    mid_point = filter_avg(mid_point)
    servo_angle = int((mid_point - 160) * 0.86)

    #yellow_tape_detect
    yellow_tape_detected_first = False
    stop_bike, yellow_tape_detected_first, yellow_mask = detect_yellow_tape(color_img, yellow_tape_detected_first)
    if stop_bike:
        print("Second yellow tape detected. Stop signal sent.")
    # visualization
    if debug == True:
        draw_lane_lines(mask_three_ch_image, birdseye_image, left_line_fit, right_line_fit, perspective_matrix,
                        mid_point)
        if yolo_label is not None and roadblock_y < express_y:
            cv2.circle(mask_three_ch_image, (roadblock_x, roadblock_y), 3, (0, 0, 255), 3)
        cv2.imshow('color_image', color_img)
        cv2.imshow('process_image', mask_three_ch_image)
        cv2.imshow('birdeye', birdseye_three_ch_image)
        cv2.imshow('yellow_mask', yellow_mask)
        cv2.waitKey(1)

    return servo_angle, roadblock_distance, stop_bike

if __name__ == '__main__':
    image = cv2.imread("./result/0.jpg")
    mid_point, result_image = post_process(image, True)
    if mid_point is not None:
        print("Mid Point:", mid_point)
    cv2.imshow("Result Image", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()