#! /usr/bin/env python3

import cv2
import rospy
import time
from sensor_msgs.msg import Image
from ringbuffer import CircularBuffer
from function import imgmsg_to_cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt


class image_view:
    def __init__(self):
        #self.input_image = rospy.Subscriber("/fps_controller/image_raw", Image, self.callback)
        self.fps_result_iamge_sub = rospy.Subscriber("/detected_class/result_image",Image, self.callback)
        self.buffer = CircularBuffer(50)
        self.bridge = CvBridge()
        self.count = 0

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            cv2.imshow("Image View", self.cv_image)

            cv2.waitKey(int(1000/30))          

        except Exception as e: 
            print(str(e))


def main():
    img_node = image_view()
    rospy.init_node('image_view', anonymous=True)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shou down")
        cv2.destroyAllWindows()

if __name__=='__main__':
    main()

