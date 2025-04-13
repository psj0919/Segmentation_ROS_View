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
model = 'deeplab'  # [ fcn | repvgg | segmenter | deeplab | RepVGG_deeplab | RepVGG_bottleneck | RepVGG_DeepLabV3+_DA_ECA]
Resize_size = 512
org_size_w = 512
org_size_h = 512
DEVICE = torch.device('cuda:0')
num_class = 21
        
CLASSES = [
    'background', 'vehicle', 'bus', 'truck', 'policeCar', 'ambulance', 'schoolBus', 'otherCar',
    'freespace', 'curb', 'safetyZone', 'roadMark', 'whiteLane',
    'yellowLane', 'blueLane', 'constructionGuide', 'trafficDrum',
    'rubberCone', 'trafficSign', 'warningTriangle', 'fence']
    
color_table = {0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0), 3: (0, 0, 128), 4: (128, 128, 0),
               5: (128, 0, 128), 6: (0, 128, 128), 7: (128, 128, 128), 8: (0, 64, 64),
               9: (64, 64, 64), 10: (0, 0, 192), 11: (192, 0, 192), 12: (0, 192, 192),
               13: (192, 192, 192), 14: (64, 128, 0), 15: (192, 0, 128), 16: (64, 128, 128),
               17: (192, 128, 128), 18: (128, 64, 0), 19: (128, 192, 0), 20: (0, 64, 128)}

PT = {'vehicle': 0.4, 'bus':0.4 ,'truck':0.4, 'policeCar':0.4,'ambulance':0.4,'schoolBus':0.4,'otherCar':0.4}

#------------------------------------------------------------------------------------------#
if model =='fcn':
    network_name = 'FCN8s' 
    if Resize_size == 256:
        if network_name == 'FCN8s' :
            weight_file = "/home/parksungjun/checkpoints/256/fcn_epochs_200_optimizer_adam_lr_0.0001_modelfcn8.pth"

        elif network_name == 'FCN16s':
            weight_file = "/home/parksungjun/checkpoints/256/fcn_epochs_200_optimizer_adam_lr_0.0001_modelfcn16.pth"

        elif network_name == 'FCN32s':
            weight_file = "/home/parksungjun/checkpoints/256/fcn_epochs_200_optimizer_adam_lr_0.0001_modelfcn32.pth"
    elif Resize_size == 512:
        if network_name == 'FCN8s' :
            weight_file = "/home/parksungjun/checkpoints/512/fcn_epochs_50_optimizer_adam_lr_0.0001_modelfcn8_max_prob_mAP.pth"

        elif network_name == 'FCN16s':
            weight_file = "/home/parksungjun/checkpoints/512/fcn_epochs_50_optimizer_adam_lr_0.0001_modelfcn16_max_prob_mAP.pth"

        elif network_name == 'FCN32s':
            weight_file = "/home/parksungjun/checkpoints/512/fcn_epochs_50_optimizer_adam_lr_0.0001_modelfcn32_max_prob_mAP.pth"
            
    sys.path.append("/home/parksungjun/model/vehicle_segmentation-main")
    from Core.demo import demo
    Detected_class = demo(network_name, Resize_size, org_size_w, org_size_h,  DEVICE, weight_file, num_class)
#------------------------------------------------------------------------------------------#    
elif model =='repvgg':
    network_name = 'pre_my'
    if Resize_size == 256:
        if network_name == 'my':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_sj_modify7x7_fcn_epochs_50_optimizer_adam_lr_0.0001_modelmy_max_prob_mAP.pth"
        elif network_name == 'pre_my':
            weight_file = "/home/parksungjun/checkpoints/256/Pretrained_RepVGG_sj_modify7x7_fcn_epochs_50_optimizer_adam_lr_0.0001_modelmy_max_prob_mAP.pth"
    elif Resize_size == 512:  
        if network_name == 'my':
            weight_file = "/home/parksungjun/checkpoints/512/RepVGG_sj_modify7x7_fcn_epochs_50_optimizer_adam_lr_0.0001_modelmy_max_prob_mAP.pth"
        elif network_name == 'pre_my':
            weight_file = "/home/parksungjun/checkpoints/512/Pretrained_RepVGG_sj_modify7x7_fcn_epochs_50_optimizer_adam_lr_0.0001_modelmy_max_prob_mAP.pth"   
                
    sys.path.append("/home/parksungjun/model/RepVGG_fcn-master")
    from Core.demo import demo
    Detected_class = demo(network_name, Resize_size, org_size_w, org_size_h,  DEVICE, weight_file, num_class)
#------------------------------------------------------------------------------------------#
elif model =='segmenter':           
    sys.path.append("/home/parksungjun/model/Segmenter-master")
    from Config.demo_config import get_config_dict
    cfg = get_config_dict()
    if Resize_size == 256:
        if cfg['dataset']['network_name'] == 'Seg-S':
            weight_file = "/home/parksungjun/checkpoints/256/Segmenter_pretrained_70_optimizer_adam_lr_0.0001_modelSeg-S_max_prob_mAP.pth"
        elif cfg['dataset']['network_name'] == 'Seg-B':
            weight_file ='/home/parksungjun/checkpoints/256/Segmenter_pretrained_70_optimizer_adam_lr_0.0001_modelSeg-B_max_prob_mAP.pth'
        elif cfg['dataset']['network_name'] == 'Seg-BP8':
            weight_file ='/home/parksungjun/checkpoints/256/Segmenter_pretrained_70_optimizer_adam_lr_0.0001_modelSeg-BP8_max_prob_mAP.pth'
        elif cfg['dataset']['network_name'] == 'Seg-L':
            weight_file ='/home/parksungjun/checkpoints/256/Segmenter_pretrained_70_optimizer_adam_lr_0.0001_modelSeg-L_max_prob_mAP.pth'        
    elif Resize_size == 512:
        if cfg['dataset']['network_name'] == 'Seg-S':
            weight_file = "/home/parksungjun/checkpoints/512/Segmenter_pretrained_70_optimizer_adam_lr_0.0001_modelSeg-S_max_prob_mAP.pth"
        elif cfg['dataset']['network_name'] == 'Seg-B':
            weight_file ='/home/parksungjun/checkpoints/512/Segmenter_pretrained_70_optimizer_adam_lr_0.0001_modelSeg-B_max_prob_mAP.pth'
        elif cfg['dataset']['network_name'] == 'Seg-L':
            weight_file ='/home/parksungjun/checkpoints/512/Segmenter_pretrained_70_optimizer_adam_lr_0.0001_modelSeg-L_max_prob_mAP.pth'     
            
    from Core.demo import demo
    Detected_class = demo(Resize_size, org_size_w, org_size_h,  DEVICE, weight_file, num_class, cfg)
#------------------------------------------------------------------------------------------#
elif model == 'deeplab':
    backbone = 'resnet50'
    sys.path.append("/home/parksungjun/model/DeepLab-master")
    if Resize_size == 256:    
        if backbone == 'resnet50':
            weight_file = "/home/parksungjun/checkpoints/256/Pretrained_resnet50_DeepLabv3+_epochs_65_optimizer_adam_lr_0.0001_modelDeepLabV3+_max_prob_mAP.pth"
        elif backbone == 'resnet101':
            weight_file = "/home/parksungjun/checkpoints/256/Pretrained_resnet101_DeepLabv3+_epochs_65_optimizer_adam_lr_0.0001_modelDeepLabV3+_max_prob_mAP.pth"
    elif Resize_size == 512:
        if backbone == 'resnet50':
            weight_file = "/home/parksungjun/checkpoints/512/Pretrained_resnet50_DeepLabv3+_epochs_65_optimizer_adam_lr_0.0001_modelDeepLabV3+_max_prob_mAP.pth"
        elif backbone == 'resnet101':
            weight_file = "/home/parksungjun/checkpoints/512/Pretrained_resnet101_DeepLabv3+_epochs_65_optimizer_adam_lr_0.0001_modelDeepLabV3+_max_prob_mAP.pth"  
              
    from Core.demo import demo
    Detected_class = demo(backbone, Resize_size, org_size_w, org_size_h,  DEVICE, weight_file, num_class)   
    
elif model == 'RepVGG_deeplab':
    backbone = 'resnet101'
    sys.path.append("/home/parksungjun/model/RepVGG_DeepLab-master")
    if Resize_size == 256:    
        if backbone == 'resnet50':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_ResNet50_28512.pth"
        elif backbone == 'resnet101':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_ResNet101_28512.pth"
    elif Resize_size == 512:
        if backbone == 'resnet50':
            weight_file = " "
        elif backbone == 'resnet101':
            weight_file = " "            
    from Core.demo import demo
    Detected_class = demo(backbone, Resize_size, org_size_w, org_size_h,  DEVICE, weight_file, num_class)         
    

elif model == 'RepVGG_bottleneck':
    backbone = 'resnet101'
    sys.path.append("/home/parksungjun/model/DeepLab_RepVGG_Bottleneck")
    if Resize_size == 256:    
        if backbone == 'resnet50':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_Bottleneck_ResNet50.pth"
        elif backbone == 'resnet101':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_Bottleneck_ResNet101.pth"
    elif Resize_size == 512:
        if backbone == 'resnet50':
            weight_file = " "
        elif backbone == 'resnet101':
            weight_file = " "            
    from Core.demo import demo
    Detected_class = demo(backbone, Resize_size, org_size_w, org_size_h,  DEVICE, weight_file, num_class)
    
    
elif model =='RepVGG_DeepLabV3+_ECA':
    backbone = 'resnet50_backbone'
    sys.path.append("/home/parksungjun/model/RepVGG_DeepLab_ECA")
    if Resize_size == 256:    
        if backbone == 'resnet50_backbone':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_ECA/ResNet50_DeepLabV3+_75456_ECA_after_backbone.pth"
            backbone = 'resnet50'
        elif backbone == 'resnet101_backbone':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_ECA/ResNet101_DeepLabV3+_75456_ECA_after_backbone.pth"
            backbone = 'resnet101'
        elif backbone == 'resnet50_bottleneck1':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_ECA/ResNet50_DeepLabV3+_75456_ECA_after_bottleneck1.pth"
            backbone = 'resnet50'
        elif backbone == 'resnet101_bottleneck1':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_ECA/ResNet101_DeepLabV3+_75456_ECA_after_bottleneck1.pth"
            backbone = 'resnet101'
        elif backbone == 'resnet50_bottleneck2':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_ECA/ResNet50_DeepLabV3+_75456_ECA_after_bottleneck2.pth"
            backbone = 'resnet50'
        elif backbone == 'resnet101_bottleneck2':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_ECA/ResNet101_DeepLabV3+_75456_ECA_after_bottleneck2.pth"
            backbone = 'resnet101'
        elif backbone == 'resnet50_bottleneck3':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_ECA/ResNet50_DeepLabV3+_75456_ECA_after_bottleneck3.pth"
            backbone = 'resnet50'
        elif backbone == 'resnet101_bottleneck3':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_ECA/ResNet101_DeepLabV3+_75456_ECA_after_bottleneck3.pth"
            backbone = 'resnet101'
        elif backbone == 'resnet50_bottleneck4':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_ECA/ResNet50_DeepLabV3+_75456_ECA_after_bottleneck4.pth"
            backbone = 'resnet50'
        elif backbone == 'resnet101_bottleneck4':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_ECA/ResNet101_DeepLabV3+_75456_ECA_after_bottleneck4.pth"                                                
            backbone = 'resnet101'                   
    from Core.demo import demo
    Detected_class = demo(backbone, Resize_size, org_size_w, org_size_h,  DEVICE, weight_file, num_class, attention='backbone')        
    

elif model =='RepVGG_DeepLabV3+_DA_ECA':
    backbone = 'resnet101_bottleneck4'
    sys.path.append("/home/parksungjun/model/RepVGG_DeepLAb_DA(Channel)_ECA")
    if Resize_size == 256:    
        if backbone == 'resnet50_backbone':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_DA(channel)_ECA/ResNet50_DeepLabV3+_75456_DA(channel)_ECA_after_backbone.pth"
            backbone = 'resnet50'
        elif backbone == 'resnet101_backbone':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_DA(channel)_ECA/ResNet101_DeepLabV3+_75456_DA(channel)_ECA_after_backbone.pth"
            backbone = 'resnet101'
        elif backbone == 'resnet50_bottleneck1':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_DA(channel)_ECA/ResNet50_DeepLabV3+_75456_DA(channel)_ECA_after_bottleneck1.pth"
            backbone = 'resnet50'
        elif backbone == 'resnet101_bottleneck1':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_DA(channel)_ECA/ResNet101_DeepLabV3+_75456_DA(channel)_ECA_after_bottleneck1.pth"
            backbone = 'resnet101'
        elif backbone == 'resnet50_bottleneck2':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_DA(channel)_ECA/ResNet50_DeepLabV3+_28512_DA(channel)_ECA_after_bottleneck2.pth"
            backbone = 'resnet50'
        elif backbone == 'resnet101_bottleneck2':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_DA(channel)_ECA/ResNet101_DeepLabV3+_28512_DA(channel)_ECA_after_bottleneck2.pth"
            backbone = 'resnet101'
        elif backbone == 'resnet50_bottleneck3':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_DA(channel)_ECA/ResNet50_DeepLabV3+_28512_DA(channel)_ECA_after_bottleneck3.pth"
            backbone = 'resnet50'
        elif backbone == 'resnet101_bottleneck3':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_DA(channel)_ECA/ResNet101_DeepLabV3+_28512_DA(channel)_ECA_after_bottleneck3.pth"
            backbone = 'resnet101'
        elif backbone == 'resnet50_bottleneck4':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_DA(channel)_ECA/ResNet50_DeepLabV3+_28512_DA(channel)_ECA_after_bottleneck4.pth"
            backbone = 'resnet50'
        elif backbone == 'resnet101_bottleneck4':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_DA(channel)_ECA/ResNet101_DeepLabV3+_28512_DA(channel)_ECA_after_bottleneck4.pth"                                                
            backbone = 'resnet101'                   
    from Core.demo import demo
    Detected_class = demo(backbone, Resize_size, org_size_w, org_size_h,  DEVICE, weight_file, num_class, attention='bottleneck4')  
   
 #---------------------------------------------------------------------------------------------------------------

class detected_class:
    def __init__(self):
        #self.input_image = rospy.Subscriber("/fps_controller/image_raw", Image, self.callback)
        self.input_image = rospy.Subscriber("video_publish/video_frames", Image, self.callback)
        self.result_image = rospy.Publisher("/detected_class/result_image", Image, queue_size=1)
        self.bridge = CvBridge()

    # for bounding box        
    def callback(self, data):
      try:     
        cv_input_image = self.bridge.imgmsg_to_cv2(data, 'rgb8')            

        #run model     
        s_time= time.time()
        output, output2= Detected_class.run(cv_input_image)        
        result_image, boundingbox = Detected_class.pred_to_rgb(output, output2, color_table, PT)
        e_time = time.time()
        print(f'total TIME: {1 / (e_time - s_time)}')
        
                
        cv_input_image = cv2.resize(cv_input_image, (512, 512), interpolation=cv2.INTER_NEAREST)                           
        #        
        result_image = cv2.addWeighted(cv_input_image, 1, result_image, 0.5, 0)
        #
        for idx, (x, y, w, h) in boundingbox:
            cv2.rectangle(result_image, (x, y), (x+w, y+h), color_table[idx], 1)
            
        result_image = cv2.resize(result_image, (640, 480), interpolation=cv2.INTER_NEAREST)                                               
        msg_result_image = self.bridge.cv2_to_imgmsg(result_image, encoding="rgb8")
 
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
