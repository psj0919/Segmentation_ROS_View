#! /usr/bin/env python3
#echo 918000000 | sudo tee /sys/devices/gpu.0/devfreq/17000000.ga10b/max_freq 
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
model = 'repvgg'  # [ fcn | repvgg | segmenter | deeplab ]
Resize_size = 256
org_size_w = 480
org_size_h = 640
DEVICE = torch.device('cuda:0')
num_class = 21
        

color_table = {0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0), 3: (0, 0, 128), 4: (128, 128, 0),
               5: (128, 0, 128), 6: (0, 128, 128), 7: (128, 128, 128), 8: (0, 64, 64),
               9: (64, 64, 64), 10: (0, 0, 192), 11: (192, 0, 192), 12: (0, 192, 192),
               13: (192, 192, 192), 14: (64, 128, 0), 15: (192, 0, 128), 16: (64, 128, 128),
               17: (192, 128, 128), 18: (128, 64, 0), 19: (128, 192, 0), 20: (0, 64, 128)}

        
#------------------------------------------------------------------------------------------#
if model =='fcn':
    network_name = 'FCN32s' 
    if network_name == 'FCN8s' :
        weight_file = "/home/parksungjun/checkpoints/fcn_epochs_200_optimizer_adam_lr_0.0001_modelfcn8.pth"

    elif network_name == 'FCN16s':
        weight_file = "/home/parksungjun/checkpoints/fcn_epochs_200_optimizer_adam_lr_0.0001_modelfcn16.pth"

    elif network_name == 'FCN32s':
        weight_file = "/home/parksungjun/checkpoints/fcn_epochs_200_optimizer_adam_lr_0.0001_modelfcn32.pth"

    sys.path.append("/home/parksungjun/model/vehicle_segmentation-main")
    from Core.demo import demo
    Detected_class = demo(network_name, Resize_size, org_size_w, org_size_h,  DEVICE, weight_file, num_class)
#------------------------------------------------------------------------------------------#    
elif model =='repvgg':
    network_name = 'my_m'
     
    if network_name == 'my':
        weight_file = "/home/parksungjun/checkpoints/RepVGG_sj_fcn_epochs_50_optimizer_adam_lr_0.0001_modelmy_max_prob_mAP.pth"
    elif network_name == 'pre_my':
        weight_file = "/home/parksungjun/checkpoints/Pretrained_RepVGG_sj_fcn_epochs_50_optimizer_adam_lr_0.0001_modelmy_max_prob_mAP.pth"
    elif network_name=="my_m":
        weight_file = "/home/parksungjun/checkpoints/_RepVGG_sj_modify_fcn_epochs_50_optimizer_adam_lr_0.0001_modelmy_max_prob_mAP.pth"                
    elif network_name=="pre_my_m":    
        weight_file = "/home/parksungjun/checkpoints/Pretrained_RepVGG_sj_modify_fcn_epochs_50_optimizer_adam_lr_0.0001_modelmy_max_prob_mAP.pth"     
    sys.path.append("/home/parksungjun/model/RepVGG_fcn-master")
    from Core.demo import demo
    Detected_class = demo(network_name, Resize_size, org_size_w, org_size_h,  DEVICE, weight_file, num_class)
#------------------------------------------------------------------------------------------#
elif model =='segmenter':           
    sys.path.append("/home/parksungjun/model/Segmenter-master")
    from Config.demo_config import get_config_dict
    cfg = get_config_dict()
    if cfg['dataset']['network_name'] == 'Seg-S':
        weight_file = "/home/parksungjun/checkpoints/Segmenter_pretrained_70_optimizer_adam_lr_0.0001_modelSeg-S_max_prob_mAP.pth"
    elif cfg['dataset']['network_name'] == 'Seg-B':
        weight_file ='/home/parksungjun/checkpoints/Segmenter_pretrained_70_optimizer_adam_lr_0.0001_modelSeg-B_max_prob_mAP.pth'
    elif cfg['dataset']['network_name'] == 'Seg-BP8':
        weight_file ='/home/parksungjun/checkpoints/Segmenter_pretrained_70_optimizer_adam_lr_0.0001_modelSeg-BP8_max_prob_mAP.pth'
    elif cfg['dataset']['network_name'] == 'Seg-L':
        weight_file ='/home/parksungjun/checkpoints/Segmenter_pretrained_70_optimizer_adam_lr_0.0001_modelSeg-L_max_prob_mAP.pth'        
    from Core.demo import demo
    Detected_class = demo(Resize_size, org_size_w, org_size_h,  DEVICE, weight_file, num_class, cfg)
#------------------------------------------------------------------------------------------#
elif model == 'deeplab':
    backbone = 'resnet101'
    sys.path.append("/home/parksungjun/model/DeepLab-master")
    if backbone == 'resnet50':
        weight_file = "/home/parksungjun/checkpoints/Pretrained_resnet50_DeepLabv3+_epochs_65_optimizer_adam_lr_0.0001_modelDeepLabV3+_max_prob_mAP.pth"
    elif backbone == 'resnet101':
        weight_file = "/home/parksungjun/checkpoints/Pretrained_resnet101_DeepLabv3+_epochs_65_optimizer_adam_lr_0.0001_modelDeepLabV3+_max_prob_mAP.pth"
    from Core.demo import demo
    Detected_class = demo(backbone, Resize_size, org_size_w, org_size_h,  DEVICE, weight_file, num_class)        
 #---------------------------------------------------------------------------------------------------------------

class detected_class:
    def __init__(self):
        self.input_image = rospy.Subscriber("/fps_controller/image_raw", Image, self.callback)
        self.result_image = rospy.Publisher("/detected_class/result_image", Image, queue_size=1)
        self.bridge = CvBridge()
        
    def callback(self, data):
      try:     
        cv_input_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')            
        s_time= time.time()     
        output = Detected_class.run(cv_input_image)
        result_image = Detected_class.pred_to_rgb(output, color_table)
        e_time = time.time()
        print(f'total TIME: {1 / (e_time - s_time)}')        
        result_image = cv2.resize(result_image, (640, 480))                           
        #        
        result_image = cv_input_image+ result_image                      
        msg_result_image = self.bridge.cv2_to_imgmsg(result_image, encoding="bgr8")
        #                        
        self.result_image.publish(msg_result_image)
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
