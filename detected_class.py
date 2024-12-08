#! /usr/bin/env python3
import cv2
import sys
import time
import torch
import torch.nn.functional as F
import rospy
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int32

# --------------------------------------------------setting model------------------------------------------------
model = 'repvgg'
Resize_size = 256
org_size_w = 480
org_size_h = 640
DEVICE = torch.device('cuda:0')
num_class = 21

if model =='fcn':
    network_name = 'FCN8s' 
    if network_name == 'FCN8s' :
        weight_file = "/home/parksungjun/Downloads/fcn_epochs_200_optimizer_adam_lr_0.0001_modelfcn8.pth"
    # when model is fcn8s, fps_controller number set '8'
    elif network_name == 'FCN16s':
        weight_file = "/home/parksungjun/Downloads/fcn_epochs_200_optimizer_adam_lr_0.0001_modelfcn16.pth"
    # when model is fcn16s, fps_controller number set '8'
    elif network_name == 'FCN32s':
        weight_file = "/home/parksungjun/Downloads/fcn_epochs_200_optimizer_adam_lr_0.0001_modelfcn32.pth"
    # when model is fcn32s, fps_controller number set '8'
    sys.path.append("/home/parksungjun/vehicle_segmentation-main")
    from Core.demo import demo
    Detected_class = demo(network_name, Resize_size, org_size_w, org_size_h,  DEVICE, weight_file, num_class)
    
    
elif model =='repvgg':
    weight_file = "/home/parksungjun/Downloads/pretrained_RepVGG_fcn_epochs_50_optimizer_adam_lr_0.0001_modela0_max_prob_mAP.pth"
    network_name = 'a0' 
    if network_name == 'a0':
        weight_file = "/home/parksungjun/Downloads/pretrained_RepVGG_fcn_epochs_50_optimizer_adam_lr_0.0001_modela0_max_prob_mAP.pth"
    elif network_name =='b0':
        weight_file = "/home/parksungjun/Downloads/pretrained_RepVGG_fcn_epochs_50_optimizer_adam_lr_0.0001_modelb0_max_prob_mAP.pth"
     
    sys.path.append("/home/parksungjun/RepVGG_fcn-master")
    from Core.demo import demo
    Detected_class = demo(network_name, Resize_size, org_size_w, org_size_h,  DEVICE, weight_file, num_class)
    # when model is repvgg, fps_controller number set '5'

# ---------------------------------------------------------------------------------------------------------------

class detected_class:
    def __init__(self):
        self.input_image = rospy.Subscriber("/fps_controller/image_raw", Image, self.callback)
        self.result_image = rospy.Publisher("/detected_class/result_image", Image, queue_size=1)
        self.bridge = CvBridge()
        
    def callback(self, data):
        try:
      
            cv_input_image_ = self.bridge.imgmsg_to_cv2(data, 'bgr8')            
            cv_input_image = cv2.resize(cv_input_image_, (256, 256))
            s_time2= time.time()     
            result_image = Detected_class.run(cv_input_image)
            e_time = time.time()
            print(f'RUN TIME: {1 / (e_time - s_time2)}')
            result_image = cv2.resize(result_image, (640, 480))            
 
            
            #        
            result_image = cv_input_image_ + result_image                      
            msg_result_image = self.bridge.cv2_to_imgmsg(result_image, encoding="bgr8")

            #                        
            self.result_image.publish(msg_result_image)
            #e_time= time.time()
            #print(f'total_fps: {1 / (e_time - s_time)}')
            
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
