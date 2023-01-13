import cv2
import numpy as np

from line_utils import *

class Measurer:
    measure_type = None

    h_min = 0
    h_max = 180
    s_min = 0
    s_max = 224
    v_min = 10
    v_max = 100
    trackbar_window_name = "Original"
    mask_window_name = "Mask"
    result_window_name = "Result"

    warp_point_top_y = 400
    warp_point_bottom_y = 450
    warp_point_left_top_x = 25
    warp_point_left_bottom_x = 0
    warp_point_right_bottom_x = 640
    warp_point_right_top_x = 620
    warp_initial_vals = [100, 100, 100, 300]

    def __init__(self, in_measure_type):
        self.measure_type = in_measure_type

    def measure_hsv(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_min = cv2.getTrackbarPos("Hue Min", self.trackbar_window_name)
        h_max = cv2.getTrackbarPos("Hue Max", self.trackbar_window_name)
        s_min = cv2.getTrackbarPos("Sat Min", self.trackbar_window_name)
        s_max = cv2.getTrackbarPos("Sat Max", self.trackbar_window_name)
        v_min = cv2.getTrackbarPos("Val Min", self.trackbar_window_name)
        v_max = cv2.getTrackbarPos("Val Max", self.trackbar_window_name)
        print(f'h_min: {h_min}, h_max: {h_max}, s_min: {s_min}, s_max: {s_max}, v_min: {v_min}, v_max: {v_max}')
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(img_hsv, lower, upper)
        imgResult = cv2.bitwise_and(img, img, mask=mask)

        cv2.moveWindow(self.mask_window_name, img.shape[1], 0)
        cv2.moveWindow(self.result_window_name, img.shape[1]*2, 0)
        cv2.imshow(self.trackbar_window_name, img)
        cv2.imshow(self.mask_window_name, mask)
        cv2.imshow(self.result_window_name, imgResult)

    def measure_hsv_ros(self, img):

        h_l = cv2.getTrackbarPos('H_lower', self.trackbar_window_name)
        s_l = cv2.getTrackbarPos('S_lower', self.trackbar_window_name)
        v_l = cv2.getTrackbarPos('V_lower', self.trackbar_window_name)
        h_u = cv2.getTrackbarPos('H_upper', self.trackbar_window_name)
        s_u = cv2.getTrackbarPos('S_upper', self.trackbar_window_name)
        v_u = cv2.getTrackbarPos('V_upper', self.trackbar_window_name)
        lower = np.array([h_l, s_l, v_l])
        upper = np.array([h_u, s_u, v_u])
                
        # print(f'h_min: {h_min}, h_max: {h_max}, s_min: {s_min}, s_max: {s_max}, v_min: {v_min}, v_max: {v_max}')
        # lower = np.array([h_min, s_min, v_min])
        # upper = np.array([h_max, s_max, v_max])        
        mask = cv2.inRange(img, lower, upper)
        imgResult = cv2.bitwise_and(img, img, mask=mask)
        
        # mask = splitColor(img, lower, upper)
        # imgResult = convertColor(mask, cv2.COLOR_HSV2BGR)
        

        # cv2.moveWindow(self.mask_window_name, img.shape[1], 0)
        # cv2.moveWindow(self.result_window_name, img.shape[1]*2, 0)
        cv2.imshow(self.trackbar_window_name, img)
        cv2.imshow(self.mask_window_name, mask)
        cv2.imshow(self.result_window_name, imgResult)
        
    def measure_warp_point(self, img):
        h, w, c = img.shape

        width_top = cv2.getTrackbarPos('Top Width', self.trackbar_window_name)
        height_top = cv2.getTrackbarPos('Top Height', self.trackbar_window_name)
        width_bottom = cv2.getTrackbarPos('Bottom Width', self.trackbar_window_name)
        height_bottom = cv2.getTrackbarPos('Bottom Height', self.trackbar_window_name)

        points = np.float32([(width_top, height_top), (w - width_top, height_top),
                             (width_bottom, height_bottom), (w - width_bottom, height_bottom)])

        print(f'left_top: {points[0]}, right_top: {points[1]}, left_bottom: {points[2]}, right_bottom: {points[3]}')

        for x in range(len(points)):
            cv2.circle(img, (int(points[x][0]), int(points[x][1])), 3, (0, 0, 255), cv2.FILLED)

        pts1 = np.float32(points)
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp = cv2.warpPerspective(img, matrix, (w, h))

        cv2.imshow(self.trackbar_window_name, img)
        cv2.imshow(self.result_window_name, img_warp)

    def initialize_hsv_trackbars(self):
        cv2.namedWindow(self.trackbar_window_name)
        cv2.createTrackbar("Hue Min", self.trackbar_window_name, self.h_min, 360, self.nothing)
        cv2.createTrackbar("Hue Max", self.trackbar_window_name, self.h_max, 360, self.nothing)
        cv2.createTrackbar("Sat Min", self.trackbar_window_name, self.s_min, 255, self.nothing)
        cv2.createTrackbar("Sat Max", self.trackbar_window_name, self.s_max, 255, self.nothing)
        cv2.createTrackbar("Val Min", self.trackbar_window_name, self.v_min, 255, self.nothing)
        cv2.createTrackbar("Val Max", self.trackbar_window_name, self.v_max, 255, self.nothing)

    def initialize_warp_trackbars(self, initial_trackbar_vals, w, h):
        cv2.namedWindow(self.trackbar_window_name)
        cv2.createTrackbar('Top Width', self.trackbar_window_name, initial_trackbar_vals[0], w // 2, self.nothing)
        cv2.createTrackbar('Top Height', self.trackbar_window_name, initial_trackbar_vals[1], h, self.nothing)
        cv2.createTrackbar('Bottom Width', self.trackbar_window_name, initial_trackbar_vals[2], w // 2, self.nothing)
        cv2.createTrackbar('Bottom Height', self.trackbar_window_name, initial_trackbar_vals[3], h, self.nothing)

    def initialize_hsv_trackbars_nwsetting(self):
        cv2.namedWindow(self.trackbar_window_name, cv2.WINDOW_GUI_EXPANDED)
        cv2.createTrackbar('H_lower', self.trackbar_window_name, 0, 360, self.nothing)
        cv2.createTrackbar('S_lower', self.trackbar_window_name, 0, 255, self.nothing)
        cv2.createTrackbar('V_lower', self.trackbar_window_name, 0, 255, self.nothing)
        cv2.createTrackbar('H_upper', self.trackbar_window_name, 180, 360, self.nothing)
        cv2.createTrackbar('S_upper', self.trackbar_window_name, 255, 255, self.nothing)
        cv2.createTrackbar('V_upper', self.trackbar_window_name, 255, 255, self.nothing)

    def nothing(self, a):
        pass