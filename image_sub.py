#! /usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import matplotlib.pyplot as plt

class image_convert:
    def __init__(self):
        self.bridge=CvBridge()
        self.color_sub=rospy.Subscriber("/camera/color/image_raw",Image, self.callback)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv2.imshow('rgb', cv_image)
            cv2.waitKey(33)
        except:
            cv_image = self.bridge.imgmsg_to_cv2(data)

def main():
    ic = image_convert()
    rospy.init_node("image_convert", anonymous=True)
    try:
        rospy.spin()
    except keyboardInterrupt:
        print("shutting down")
        cv2.destroyAllWindows()

if __name__=='__main__':
    main()
