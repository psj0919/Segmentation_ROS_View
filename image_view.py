#! /usr/bin/env python3

import cv2
import rospy
from sensor_msgs.msg import Image
from ringbuffer import CircularBuffer
from function import imgmsg_to_cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt


class image_view:
    def __init__(self):
        self.input_image = rospy.Subscriber("/fps_controller/image_raw", Image, self.callback)
        self.fps_result_iamge_sub = rospy.Subscriber("/detected_class/result_image",Image, self.callback_result)
        self.buffer = CircularBuffer(50)
        self.buffer2 = CircularBuffer(50)
        self.bridge = CvBridge()

    def callback(self, data):
        try: 
            self.cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            self.buffer.enqueue(self.cv_image)
            cv2.imshow('Image View', self.buffer.dequeue())
            cv2.waitKey(33)
        except CvBridgeError as e:
            print(e)

    def callback_result(self, data):
        try:

            self.cv_image2 = self.bridge.imgmsg_to_cv2(data, 'rgba8')
            self.buffer2.enqueue(self.cv_image2)
            cv2.imshow('Result Image View', self.buffer2.dequeue())
            cv2.waitKey(33)
        except:
            print("Result_Image View Error")	


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
