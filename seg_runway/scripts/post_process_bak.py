import numpy as np
import cv2
import random as rng
import math

color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))

# to be used, if this angle is too different from last time,
# do some process, like keep last time angle, or do average
last_angle = 0
this_angle = 0
first_detect = False


def post_process(image, debug):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # this input is np.float32 type image, but the findContours function can only process np.uint8 image
    mask = np.zeros((gray_image.shape[0], gray_image.shape[1]), dtype=np.uint8)  # new a uint8 image
    mask[:] = gray_image[:] * 255  # the cnn out image value is in [0-1]
    ret_val, threshold_image = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing_image = np.zeros((threshold_image.shape[0], threshold_image.shape[1], 3), dtype=np.uint8)
    line_points = []
    for i in range(len(contours)):
        cv2.drawContours(drawing_image, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
        boundbox = cv2.boundingRect(contours[i])
        x, y, w, h = boundbox
        # roi is not used
        roi = threshold_image[y: y + h, x: x + w]
        half_roi = threshold_image[y + int(h / 2): y + h, x: x + w]

        for row in range(h):
            row_num = row + y
            for point in contours[i]:
                if point[0][1] == row_num:
                    line_points.append([point[0][0], point[0][1]])

        output = cv2.fitLine(np.array(line_points), cv2.DIST_L2, 0, 0.01, 0.01)
        k = output[1] / output[0]
        b = output[3] - k * output[2]

        angle = math.atan(k) * 180 / math.pi

        if debug:
            for point in contours[i]:
                x = point[0][0]
                y = int(k * x + b)
                cv2.circle(drawing_image, (x, y), 1, (255, 255, 0), 1)
            cv2.putText(drawing_image, "angle: " + str(int(angle)), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 255), 1)
            cv2.imshow("roi", roi)
            cv2.imshow("half_roi", half_roi)

    if debug:
        cv2.imshow("draw", drawing_image)
        cv2.waitKey(0)


if __name__ == '__main__':
    image = cv2.imread("./result/0.jpg")
    image.astype(np.uint8)
    print(image.dtype)
    post_process(image, True)
