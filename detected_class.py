#! /usr/bin/env python3
import cv2
import sys
import time
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
weight_file = "/home/parksungjun/Downloads/fcn_epochs_200_optimizer_adam_lr_0.0001_modelfcn32.pth"
GPU_ID = '0'
DEVICE = torch.device('cuda:0')
network_name = 'FCN32s' 
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
            cv_input_image = cv2.resize(cv_input_image, (256, 256))
            s_time= time.time()
            result_image = Detected_class.run(cv_input_image)
            e_time= time.time()
            #print(f'RUN TIME: {1 / (e_time - s_time)}')
            

            #result_image = result_image.to('cpu', non_blocking=True)
            #result_image = result_image.numpy().astype(np.uint8)

            
            cv_input_image = cv2.resize(cv_input_image, (640, 480))
            s_time= time.time()
            result_image = cv2.resize(result_image.to('cpu').numpy().astype(np.uint8), (640, 480))
            e_time= time.time()
            #print(f'Cpu_TIME: {1 / (e_time - s_time)}')            
            result_image = cv_input_image + result_image
            
            
            result_image = self.bridge.cv2_to_imgmsg(result_image, encoding="bgr8")
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
