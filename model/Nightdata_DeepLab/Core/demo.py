import cv2
import time
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model.RepVGG_ResNet_deeplabv3plus import DeepLab
from Preprocessing_model.retinexformer import *
from Preprocessing_model.CIDNet.CIDNet import *
from Preprocessing_model.Preprocessing import *
from Preprocessing_model.Gamma_correction_sj import *
from Preprocessing_model.Gamma_correction_sj2 import *
from Preprocessing_model.AGC import *

CLASSES = [
    'background', 'vehicle', 'bus', 'truck', 'policeCar', 'ambulance', 'schoolBus', 'otherCar',
    'freespace', 'curb', 'safetyZone', 'roadMark', 'whiteLane',
    'yellowLane', 'blueLane', 'constructionGuide', 'trafficDrum',
    'rubberCone', 'trafficSign', 'warningTriangle', 'fence']

class demo:
    def __init__(self, backbone, resize_size, org_size_w, org_size_h, device, weight_path, num_class):
        self.backbone = backbone
        self.resize_size = resize_size
        self.org_size_w = org_size_w
        self.org_size_h = org_size_h
        self.scale_x = org_size_h / resize_size
        self.scale_y = org_size_w / resize_size        
        self.device = device
        self.weight_path = weight_path
        self.num_class = num_class
        self.model = self.load_network()
        #self.preprocess_model = self.get_retinexformer()
        self.preprocess_model = self.get_cidnet()
        #self.preprocess_gamma = self.get_gamma_correction_sj2()
        self.gamma = gamma_correction_sj().to(self.device).eval().half()
        self.load_weight()
        
        
    def load_network(self):
        pretrain = False
        model = DeepLab(num_classes=21, backbone=self.backbone,
                        output_stride=16, sync_bn=False, freeze_bn=False, pretrained=False, deploy=True) 
                        
        return model.eval().to(self.device)

    def get_retinexformer(self):
        model = RetinexFormer(stage=1, n_feat=40, num_blocks=[1, 2, 2])
        path = '/home/parksungjun/Night_checkpoints/retinexformer.pth'
        ckpt = torch.load(path, map_location=self.device)
        try:
            model.load_state_dict(ckpt, strict = True)
        except:
            print("Error")
            
        return model.eval().to(self.device)
    
    def get_cidnet(self):
        model = CIDNet()
        path = '/home/parksungjun/Night_checkpoints/CIDNet.pth'
        
        ckpt = torch.load(path, map_location=self.device)
        try:
            model.load_state_dict(ckpt, strict = True)
        except:
            print("Error")
            
        return model.to(self.device)
    
    def get_gamma_correction_sj2(self):
        model = gamma_correction()
        path = '/home/parksungjun/Night_checkpoints/gamma_correction_sj2'

        ckpt = torch.load(path, map_location=self.device)
        try:
            model.load_state_dict(ckpt, strict = True)
        except:
            print("Error")
            
        return model.to(self.device)        
        
    def load_weight(self):
        file_path = self.weight_path
        assert os.path.exists(file_path), f'There is no checkpoints file!'
        print("Loading saved weighted {}".format(file_path))
        ckpt = torch.load(file_path, map_location=self.device)
        
        self.model.load_state_dict(ckpt, strict=True)  # load weights
        
        
    @torch.no_grad()
    def run(self, data):
        # pre-processing 
        #data = adaptive_gamma_correction(data)   # [histogram_equal | clahe | gammacorrection]
        img = self.transform(data).to(self.device)
        #img, gamma = self.preprocess_gamma(img) # [gamma_correction_sj | gamma_correction_sj2]
        # model
        with torch.cuda.amp.autocast():
            img = self.preprocess_model(img)
            output = self.model(img)
        output = torch.softmax(output[0], dim=0)
        output = torch.argmax(output, dim=0).to(torch.int8)
        output = output.cpu().numpy()        
        return output


    def transform(self, img):
        img = cv2.resize(img, (self.resize_size, self.resize_size))
        # Preprocessing [AGC, retinex_MSR, retinex_MSRCR]
        #img = histogram_equal(img)    
        #img = adaptive_gamma_correction(img)    
        #img = clahe(img)        
        #
        #img = retinex_MSRCR(img)
        img = torch.from_numpy(img.transpose(2, 0, 1)) / 255.0
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225])
        #img = normalize(img)
        return img.unsqueeze(0)
        

    def pred_to_rgb(self, pred, color_table, PT):
        
        pred_rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
        
        for i in range(len(CLASSES)):
            if CLASSES[i] == 'constructionGuide' or CLASSES[i] == 'warningTriangle':
                pass
            else:
                pred_rgb[pred ==i] = np.array(color_table[i])  
        e_time = time.time()
        #print(1 / (e_time - s_time))              
        return pred_rgb
        
        
    def make_bounding_box(self, pred, PT):
        vehicle_class_id = [1, 2, 3, 4, 5, 6, 7]

        bounding_box = []
        
        for class_id in vehicle_class_id:
            class_prob_map = pred[class_id]            
            class_mask = (class_prob_map >= PT[CLASSES[class_id]]).astype(np.uint8)
                
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h <40:
                    continue
                bounding_box.append((class_id, (int(x*self.scale_x), int(y*self.scale_y), int(w*self.scale_x), int(h*self.scale_y))))  
                
        return bounding_box
