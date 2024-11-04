#! /usr/bin/env python3
import cv2
import sys

sys.path.append("/home/parksungjun/vehicle_segmentation-main")
import torch
from cv_bridge import CvBridge, CvBridgeError
import rospy
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from Core.demo import demo

Resize_size = 256
org_size_w = 480
org_size_h = 640
weight_file = "/home/parksungjun/Downloads/fcn_epochs_200_optimizer_adam_lr_0.0001_modelfcn8.pth"
GPU_ID = '0'
DEVICE = torch.device('cuda:0')
network_name = 'FCN8s' 
num_class = 21

Detected_class = demo(network_name, Resize_size, org_size_w, org_size_h,  DEVICE, weight_file, num_class)

class detected_class:
    def __init__(self):
        self.input_image = rospy.Subscriber("/fps_controller/image_raw", Image, self.callback)
        self.result_image = rospy.Publisher("/detected_class/result_image", Image, queue_size=10)
        self.bridge = CvBridge()
        
    def callback(self, data):
        try:
            cv_input_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            result_image = Detected_class.run(cv_input_image)
            result_image = self.bridge.cv2_to_imgmsg(result_image, encoding="rgba8")
            self.result_image.publish(result_image)

        except CvBridgeError as e:
            print(e)

def main():
    class_detector = detected_class()
    rospy.init_node('class_detector', anonymous=True)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shut down")
        cv2.destroyWindow()

if __name__=='__main__':
    main()
