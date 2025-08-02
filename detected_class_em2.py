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
model = 'RepVGG_deeplab'  # [ fcn | repvgg | segmenter | deeplab | RepVGG_deeplab | RepVGG_DeepLabV3+_DA_ECA | Swin-T]
Resize_size = 512
org_size_w = 512
org_size_h = 512
DEVICE = torch.device('cuda:0')
num_class = 21
bounding_box = False
        
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
if model == 'deeplab':
    backbone = 'resnet101'
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
    Detected_class = demo(backbone, Resize_size, org_size_w, org_size_h,  DEVICE, weight_file, num_class, bounding_box)   
    
elif model == 'RepVGG_deeplab':
    backbone = 'resnet101'
    sys.path.append("/home/parksungjun/model/RepVGG_DeepLab-master")
    if Resize_size == 256:    
        if backbone == 'resnet50':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_ResNet50_75456.pth"
        elif backbone == 'resnet101':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_ResNet101_75456.pth"
    elif Resize_size == 512:
        if backbone == 'resnet50':
            weight_file = "/home/parksungjun/checkpoints/512/512_RepBlock_ResNet50_DeepLabV3+.pth"
        elif backbone == 'resnet101':
            weight_file = "/home/parksungjun/checkpoints/512/512_RepBlock_ResNet101_DeepLabV3+.pth"            
    from Core.demo import demo
    Detected_class = demo(backbone, Resize_size, org_size_w, org_size_h,  DEVICE, weight_file, num_class, bounding_box)         

elif model =='RepVGG_DeepLabV3+_DA_ECA':
    backbone = 'resnet50_backbone'
    sys.path.append("/home/parksungjun/model/RepVGG_DeepLAb_DA(Channel)_ECA")
    if Resize_size == 256:    
        if backbone == 'resnet50_backbone':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_DA(channel)_ECA/ResNet50_DeepLabV3+_75456_DA_ECA_bottleneck1+bottleneck2+bottleneck3.pth"
            backbone = 'resnet50'
        elif backbone == 'resnet101_backbone':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_DA(channel)_ECA/ResNet101_DeepLabV3+_75456_DA_ECA_bottleneck1+bottleneck2+bottleneck3+bottleneck4.pth"
            backbone = 'resnet101'
        elif backbone == 'resnet50_bottleneck1':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_DA(channel)_ECA/ResNet50_DeepLabV3+_75456_DA(channel)_ECA_after_bottleneck1.pth"
            backbone = 'resnet50'
        elif backbone == 'resnet101_bottleneck1':
            weight_file = "/home/parksungjun/checkpoints/256/RepVGG_DeepLabV3+_DA(channel)_ECA/ResNet101_DeepLabV3+_75456_DA(channel)_ECA_after_bottleneck1.pth"
            backbone = 'resnet101'
               
    from Core.demo import demo
    Detected_class = demo(backbone, Resize_size, org_size_w, org_size_h,  DEVICE, weight_file, num_class, attention='backbone')  

elif model == "Swin2-T":
    sys.path.append("/home/parksungjun/model/Swin_Transformer2-master")
    weight_file = '/home/parksungjun/Downloads/512_Pretrain_Weekly_Swin_Transformer-T.pth'
    #weight_file = '/home/parksungjun/Night_checkpoints/Swin_Transformer2/Pretrained_Night_Swin_Transformer-S.pth'
    #weight_file = '/home/parksungjun/Night_checkpoints/Swin_Transformer2/Pretrained_Night_Swin_Transformer-B.pth'
    #weight_file = '/home/parksungjun/Night_checkpoints/Swin_Transformer2/Pretrained_Night_Swin_Transformer-L.pth'            
    Resize_size = 512
    org_size_w = 512
    org_size_h = 512
    model_name = model
    from Core.demo import demo
    Detected_class = demo(Resize_size, org_size_w, org_size_h,  DEVICE, weight_file, num_class, model_name)
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------

class detected_class:
    def __init__(self):
        #self.input_image = rospy.Subscriber("/fps_controller/image_raw", Image, self.callback)
        self.input_image = rospy.Subscriber("video_publish/video_frames", Image, self.callback)
        self.result_image = rospy.Publisher("/detected_class_encoder/result_image", Image, queue_size=1)
        #self.result_image = rospy.Subscriber("/detected_class_encoder/result_image", Image, self.callback)
        self.bridge = CvBridge()
        
        self.total_time = 0
        self.count = 0

    # Non-bounding box
    def callback(self, data):
      try:     
        cv_input_image = self.bridge.imgmsg_to_cv2(data, 'rgb8')            

        #run model     
   
        s_time= time.time()
        output, output2 = Detected_class.run(cv_input_image)        
        result_image = Detected_class.pred_to_rgb(output2, PT)            
        e_time = time.time()
        print(f'total TIME: {1 / (e_time - s_time)}')
                
        #cv_input_image = cv2.resize(cv_input_image, (512, 512), interpolation=cv2.INTER_NEAREST)                           
        #        
        #result_image = cv2.addWeighted(cv_input_image, 1, result_image, 0.5, 0)
        #            
        #result_image = cv2.resize(result_image, (640, 480), interpolation=cv2.INTER_NEAREST)                                               
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
