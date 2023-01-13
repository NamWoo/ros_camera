#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from line_utils import *

h_l = 0
s_l = 100
v_l = 120
h_u = 25
s_u = 255
v_u = 160

lower = np.array([h_l, s_l, v_l])
upper = np.array([h_u, s_u, v_u])

def line_dectecting(cv_image):
    
    cv_image = convertColor(cv_image, cv2.COLOR_BGR2HSV)
    cv_image = splitColor(cv_image, lower, upper)
    cv_image = convertColor(cv_image, cv2.COLOR_HSV2BGR)
    
    cv_image = cannyEdge(cv_image, 100, 200)
    
    lines = houghLinesP(cv_image, 1, np.pi/180, 100, 10, 50)
    cv_image = drawHoughLinesP(cv_image, lines)


    return cv_image



class Measurer:
    
    measure_type = None

    h_min = 0
    h_max = 0
    s_min = 0
    s_max = 0
    v_min = 10
    v_max = 180
    trackbar_window_name = "Original"
    mask_window_name = "Mask"
    result_window_name = "Result"

    warp_point_top_y = 400
    warp_point_bottom_y = 650
    warp_point_left_top_x = 305
    warp_point_right_top_x = 1280-305
    warp_point_left_bottom_x = 70
    warp_point_right_bottom_x = 1280-70
    warp_initial_vals = [100, 100, 100, 300]

    def __init__(self, in_measure_type='hsv'):
        self.measure_type = in_measure_type

    def measure_hsv(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_min = cv2.getTrackbarPos("Hue Min", self.trackbar_window_name)
        h_max = cv2.getTrackbarPos("Hue Max", self.trackbar_window_name)
        s_min = cv2.getTrackbarPos("Sat Min", self.trackbar_window_name)
        s_max = cv2.getTrackbarPos("Sat Max", self.trackbar_window_name)
        v_min = cv2.getTrackbarPos("Val Min", self.trackbar_window_name)
        v_max = cv2.getTrackbarPos("Val Max", self.trackbar_window_name)
        # print(f'h_min: {h_min}, h_max: {h_max}, s_min: {s_min}, s_max: {s_max}, v_min: {v_min}, v_max: {v_max}')
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
        # # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # h_min = cv2.getTrackbarPos("Hue Min", self.trackbar_window_name)
        # h_max = cv2.getTrackbarPos("Hue Max", self.trackbar_window_name)
        # s_min = cv2.getTrackbarPos("Sat Min", self.trackbar_window_name)
        # s_max = cv2.getTrackbarPos("Sat Max", self.trackbar_window_name)
        # v_min = cv2.getTrackbarPos("Val Min", self.trackbar_window_name)
        # v_max = cv2.getTrackbarPos("Val Max", self.trackbar_window_name)
        
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

        # print(f'left_top: {points[0]}, right_top: {points[1]}, left_bottom: {points[2]}, right_bottom: {points[3]}')

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
    


class LaneDetector:
    def __init__(self):
        self.h_min = 0      # hue(색상) 최소값
        self.h_max = 0    # hue(색상) 최대값
        self.s_min = 0      # saturation(채도) 최소값
        self.s_max = 0    # saturation(채도) 최대값
        self.v_min = 100     # value(명도) 최소값
        self.v_max = 180    # value(명도) 최대값

        self.warp_point_top_y = 400             # 워핑할 이미지의 상단 y 좌표
        self.warp_point_bottom_y = 450          # 워핑할 이미지의 하단 y 좌표
        self.warp_point_left_top_x = 25         # 워핑할 이미지의 왼쪽 상단 x 좌표
        self.warp_point_left_bottom_x = 0       # 워핑할 이미지의 왼쪽 하단 x 좌표
        self.warp_point_right_bottom_x = 640    # 워핑할 이미지의 오른쪽 하단 x 좌표
        self.warp_point_right_top_x = 620       # 워핑할 이미지의 오른쪽 상단 x 좌표

        self.points = (np.array([[self.warp_point_left_top_x, self.warp_point_top_y],
                                 [self.warp_point_right_top_x, self.warp_point_top_y],
                                 [self.warp_point_left_bottom_x, self.warp_point_bottom_y],
                                 [self.warp_point_right_bottom_x, self.warp_point_bottom_y]]))

        self.lane_window_name = "LaneWindow"
        self.sliding_window_name = "SlidingWindow"

    def warp_img(self, in_img, points, w, h):
        pts1 = np.float32(points)
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp = cv2.warpPerspective(in_img, matrix, (w, h))

        return img_warp

    def filter_img(self, in_img):
        img_filtered = cv2.Canny(in_img, 50, 100)

        img_sobel_x = cv2.Sobel(img_filtered, cv2.CV_64F, 1, 0, ksize=3)
        img_sobel_x = cv2.convertScaleAbs(img_sobel_x)

        img_sobel_y = cv2.Sobel(img_filtered, cv2.CV_64F, 0, 1, ksize=3)
        img_sobel_y = cv2.convertScaleAbs(img_sobel_y)

        img_filtered = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 2, 0);

        kernel = np.ones((3, 3), np.uint8)
        img_filtered = cv2.dilate(img_filtered, kernel, iterations = 1)

        return img_filtered

    def filter_img_hsv(self, in_img, h_min, h_max, s_min, s_max, v_min, v_max):
        img_hsv = cv2.cvtColor(in_img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([h_min, s_min, v_min])
        upper_white = np.array([h_max, s_max, v_max])
        img_filtered = cv2.inRange(img_hsv, lower_white, upper_white)
        img_filtered = cv2.GaussianBlur(img_filtered, (3, 3), 0)

        return img_filtered

    def get_hist(self, in_img, img_y_top=0):
        hist = np.sum(in_img[img_y_top:, :], axis=0)
        return hist

    def sliding_window(self, in_img, num_windows=1, margin=50, minpix=1, draw_windows=True):
        global x_center, x_center_gap
        center_margin = 40
        out_img = np.dstack((in_img, in_img, in_img))
        histogram = self.get_hist(in_img)

        # find peaks of left and right halves
        first_occur_idx = np.where(histogram > 0)[0][0]
        # x_base = np.argmax(histogram)
        x_base = first_occur_idx

        window_height = int((in_img.shape[0] / num_windows))

        # Identify the x and y positions of all nonzero_indices pixels in the image
        nonzero_inds = in_img.nonzero()
        nonzero_y = np.array(nonzero_inds[0])
        nonzero_x = np.array(nonzero_inds[1])

        # Current positions to be updated for each window
        x_current = x_base

        # 좌/우측 레인의 픽셀값 리스트
        lane_inds = []

        # Step through the windows one by one
        for num_window in range(num_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = in_img.shape[0] - (num_window + 1) * window_height
            win_y_high = in_img.shape[0] - num_window * window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin

            # draw window
            if draw_windows:
                cv2.rectangle(out_img,(win_x_low,win_y_low),(win_x_high,win_y_high),
                              (100,255,255), 3)
                cv2.putText(out_img, str(num_window), (win_x_high-40, win_y_low+((win_y_high-win_y_low)//2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            # window 범위 내에 있는 nonzero_indices pixel 추출
            left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                         (nonzero_x >= win_x_low) &  (nonzero_x < win_x_high)).nonzero()[0]

            # 추출한 픽셀 인덱스 추가
            lane_inds.append(left_inds)

            # found > minpix pixels 이면 다음 윈도우의 중심점 평균 재계산
            if len(left_inds) > minpix:
                x_current = int(np.mean(nonzero_x[left_inds]))

        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)

        # 좌/우 레인의 픽셀 위치 추출
        x = nonzero_x[lane_inds]
        left_y = nonzero_y[lane_inds]

        ploty = np.linspace(0, in_img.shape[0] - 1, in_img.shape[0])
        left_fitx = np.zeros(in_img.shape[0])

        left_fit = np.zeros(3)
        x_center = center_margin

        if len(x) > 0:
            # 2차함수 근사
            # left_fit = np.polyfit(left_y, x, 2)
            # left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]

            # 1차함수 근사
            left_fit = np.polyfit(left_y, x, 1)
            left_fitx = left_fit[0]*ploty + left_fit[1]

            out_img[nonzero_y[lane_inds], nonzero_x[lane_inds]] = [0, 0, 255]
            x_center = left_fitx + center_margin

        return out_img, left_fitx, left_fit, x_center

    def detect(self, in_img, draw_windows=True):
        img_filtered = self.filter_img_hsv(in_img, self.h_min, self.h_max, self.s_min, self.s_max, self.v_min, self.v_max)
        img_filtered = self.filter_img(img_filtered)
        h, w = img_filtered.shape
        img_warp_filtered = self.warp_img(img_filtered, self.points, w, h-self.warp_point_top_y)
        img_lane, lane, lane_poly, x_center = self.sliding_window(img_warp_filtered)
        # x_center = 9
        for i in range(x_center.size):
            cv2.circle(in_img, (int(x_center[i]), i + self.warp_point_top_y), 2, (255, 0, 0), -1)

        # cv2.imshow(self.lane_window_name, img_filtered)


        if draw_windows:
            cv2.moveWindow(self.sliding_window_name, in_img.shape[1], 0)
            cv2.imshow(self.lane_window_name, in_img)
            cv2.imshow(self.sliding_window_name, img_lane)

        return in_img, x_center
