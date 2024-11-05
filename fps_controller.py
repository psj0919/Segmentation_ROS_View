#!/usr/bin/env python3

import cv2
import sys

import rospy
from sensor_msgs.msg import Image
from ringbuffer import CircularBuffer



class fps_controller:
    def __init__(self):
        self.color_image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        self.result_image_pub = rospy.Publisher("/fps_controller/image_raw",Image, queue_size=1)

        self.count = 0

    def callback(self, data):
        try:
            self.count += 1
            if self.count % 75 == 0:
                self.result_image_pub.publish(data)
            elif self.count >= 100:
                self.count = 0
        except:
            print("Error fps_controller")


def main():
    img_node = fps_controller()
    rospy.init_node('fps_controller', anonymous=True)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shut down")
        cv2.destroyAllWindows()


if __name__== '__main__':
    main()
