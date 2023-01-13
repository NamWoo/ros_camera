#!/usr/bin/env python
# -*- coding: utf-8 -*-
##from https://github.com/NamWoo

from __future__ import print_function
#import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from line_functions import *
# from measurer_ros import *

           
setArgs = True

class image_converter:
    def __init__(self):
    # self.image_pub = rospy.Publisher("image_topic_2",Image)
        ros_topic = "rgb"
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(ros_topic, Image, self.callback)
        cv2.namedWindow("output", cv2.WINDOW_GUI_EXPANDED)
        rospy.loginfo("ros camera : %s", ros_topic)
        
        self.cnt = 0

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        if self.cnt == 0:
            h, w, c = cv_image.shape
            print(h, w)
            self.cnt += 1

        # cv_image = cv2.resize(cv_image, (720, 640))


        img_detect, x_center = detector.detect(cv_image)
        # cv_image = line_dectecting(cv_image)
        
        # measurer.measure_hsv_ros(cv_image)
        # measurer.measure_warp_point(cv_image)
        
        # cv2.imshow("output", cv_image)
        cv2.waitKey(3)

    # try:
    #   self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    # except CvBridgeError as e:
    #   print(e)


def main(args):
    
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

def nothing(x):
    pass

if __name__ == '__main__':
    
    # measurer = Measurer()
    # measurer.initialize_hsv_trackbars_nwsetting()
    detector = LaneDetector()

    main(sys.argv)
