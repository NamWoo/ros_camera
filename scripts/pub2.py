#!/usr/bin/env python

##from https://github.com/NamWoo

from __future__ import print_function
#import roslib
#roslib.load_manifest('my_package')
import sys, time
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def show_camera():
    pipeline_1 = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)15/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'

    #pipeline_2 = "v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, framerate=15/1 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

    #pipeline_3 = "v4l2src device=/dev/video0 ! video/x-h264, width=1280, height=720, framerate=30/1, format=H264 ! avdec_h264 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink sync=false"

        
    cap = cv2.VideoCapture(pipeline_1, cv2.CAP_GSTREAMER)
    width = 640
    height = 480
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)
    now = time.time()
    save_name = time.strftime('%m%d_%H%M%S', time.localtime(now))
    savepath = '/root/data/temp/{}.avi'.format(save_name)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('get :', frame_width, frame_height, width, height)
    
    # out = cv2.VideoWriter(savepath, fourcc, fps, (width, height), True)
    
    
    rospy.init_node('rgb_pub', anonymous=True)
    rgb_pub = rospy.Publish('rgb', Image, queue_size=1)
    bridge = CvBridge()
    
    
    if cap.isOpened():
        try:
            while True:
                ret_val, frame = cap.read()
                rgb_pub.publish(bridge.cv2_to_imgmsg(frame, 'bgr8'))
                # out.write(frame)

                # cv2.imshow('frame', frame)
                # keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                # if keyCode == 27 or keyCode == ord('q'):
                #     break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print('end')
    else:
        print("Error: Unable to open camera")


if __name__ == '__main__':

    show_camera()
    