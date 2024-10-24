#! /usr/bin/env python3

import cv2
import rospy
from sensor_msgs.msg import Image
from ringbuffer import CircularBuffer
from function import imgmsg_to_cv2
from cv_bridge import CvBridge


class image_view:
    def __init__(self):
        self.fps_result_iamge_sub = rospy.Subscriber("/fps_controller/image_raw",Image, self.callback)
        self.buffer = CircularBuffer(50)
        self.bridge = CvBridge()

    def callback(self, data):
        try: 
            self.cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            self.buffer.enqueue(self.cv_image)
            cv2.imshow('Image View', self.buffer.dequeue())
            cv2.waitKey(33)
        except:
            print("Image View Error")


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

