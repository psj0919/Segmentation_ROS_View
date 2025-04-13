import cv2
import time
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model.deeplabv3plus import DeepLab

CLASSES = [
    'background', 'vehicle', 'bus', 'truck', 'policeCar', 'ambulance', 'schoolBus', 'otherCar',
    'freespace', 'curb', 'safetyZone', 'roadMark', 'whiteLane',
    'yellowLane', 'blueLane', 'constructionGuide', 'trafficDrum',
    'rubberCone', 'trafficSign', 'warningTriangle', 'fence']
    
Vehicle_CLASSES = ['vehicle', 'bus', 'truck', 'policeCar', 'ambulance', 'schoolBus', 'otherCar']

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
        self.load_weight()
        
        
    def load_network(self):
        pretrain = False
        model = DeepLab(num_classes=self.num_class, backbone=self.backbone,
                        output_stride=16, sync_bn=False, freeze_bn=False, pretrained=pretrain) 
                        
        return model.eval().to(self.device).half()

    def load_weight(self):
        file_path = self.weight_path
        assert os.path.exists(file_path), f'There is no checkpoints file!'
        print("Loading saved weighted {}".format(file_path))
        ckpt = torch.load(file_path, map_location=self.device)
        
        resume_state_dict = ckpt['model'].state_dict()


        self.model.load_state_dict(resume_state_dict, strict=True)  # load weights
        
        
    @torch.no_grad()
    def run(self, data):
        # pre processing
        img = self.transform(data).to(self.device).half()
        # model
        with torch.cuda.amp.autocast():
            output = self.model(img)
        output = torch.softmax(output[0], dim=0)
        output2 = torch.argmax(output, dim=0).to(torch.int8).cpu().numpy()
        
        output = output.cpu().numpy()        

        return output, output2


    def transform(self, img):
        img = cv2.resize(img, (self.resize_size, self.resize_size))    
        img = torch.from_numpy(img.transpose(2, 0, 1)) / 255.0
        return img.unsqueeze(0)
        

    def pred_to_rgb(self, pred, pred2, color_table, PT):
        vehicle_class_id = [1, 2, 3, 4, 5, 6, 7]
        

        pred_rgb = np.zeros((*pred2.shape, 3), dtype=np.uint8)
        
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


        for i in range(len(CLASSES)):
            if CLASSES[i] == 'constructionGuide' or CLASSES[i] == 'warningTriangle':
                pass
            else:
                pred_rgb[pred2 ==i] = np.array(color_table[i])  
        e_time = time.time()
        #print(1 / (e_time - s_time))              
        return pred_rgb, bounding_box
