import cv2
import numpy as np

# 输入一堆直线，返回每条直线的斜率和截距
def get_lines_fangcheng(lines):
    lines_fangcheng = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 != x2:  # 避免除以零
            k = (y2 - y1) / (x2 - x1)  # 求出直线的斜率
            b = y1 - k * x1  # 求出直线的截距
            lines_fangcheng.append((k, b))
    return lines_fangcheng

# 主函数
def main():
    cap = cv2.VideoCapture("videos/20230831.mp4")  # 更改为视频文件的路径
    if not cap.isOpened():
        print("Could not open video file...")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dst = cv2.medianBlur(frame, 3)  # 对噪声比较敏感，故须先做中值滤波

        gray_src = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)  # 灰度化

        edges = cv2.Canny(gray_src, 20, 30, 3)  # 检测边缘

        plines = []  # 定义一个存放直线信息的向量
        a = []  # 存储斜率和截距

        # Hough直线检测
        plines = cv2.HoughLinesP(edges, 1, np.pi / 180, 70, minLineLength=100, maxLineGap=50)

        if plines is not None:
            a = get_lines_fangcheng(plines)
            for i in range(len(plines)):
                x1, y1, x2, y2 = plines[i][0]
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                len_squared = (x2 - x1) ** 2 + (y2 - y1) ** 2
                line_length = np.sqrt(len_squared)  # 计算线段的长度

                if (a[i][0] > 0.25 and a[i][0] < 3) or (a[i][0] < -0.3 and a[i][0] > -1.8):  # 过滤斜率不对的直线
                    if y1 > 200 and line_length > 150 and a[i][0] > 0:  # 过滤
                        if x1 < 500:  # 沿直线调整起点
                            p = (500 - x1) * a[i][0]
                            x1 = 500
                            y1 += p
                        if y1 < 500:
                            p = (500 - y1) / a[i][0]
                            y1 = 500
                            x1 += p
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2, cv2.LINE_AA)  # 绘制符合的直线
                    elif line_length > 220 and a[i][0] < 0:  # 过滤直线
                        if x2 > 600:  # 沿直线调整终点
                            p = (600 - x2) * a[i][0]
                            x2 = 600
                            y2 += p
                        if y2 < 500:
                            p = (500 - y2) / a[i][0]
                            y2 = 500
                            x2 += p
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2, cv2.LINE_AA)  # 绘制符合的直线

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
