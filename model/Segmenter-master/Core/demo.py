import cv2
import time
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from copy import deepcopy
from Segmenter.factory import create_segmenter
import torch.nn.functional as F


CLASSES = [
    'background', 'vehicle', 'bus', 'truck', 'policeCar', 'ambulance', 'schoolBus', 'otherCar',
    'freespace', 'curb', 'safetyZone', 'roadMark', 'whiteLane',
    'yellowLane', 'blueLane', 'constructionGuide', 'trafficDrum',
    'rubberCone', 'trafficSign', 'warningTriangle', 'fence']


class demo:
    def __init__(self, resize_size, org_size_w, org_size_h, device, weight_path, num_class, cfg):
        self.cfg = cfg
        self.resize_size = resize_size
        self.org_size_w = org_size_w
        self.org_size_h = org_size_h
        self.device = device
        self.weight_path = weight_path
        self.num_class = num_class
        self.model = self.load_network()
        self.load_weight()


    def load_network(self):
        model = create_segmenter(self.cfg)
        return model.to(self.device)


    def load_weight(self):
        file_path = self.weight_path
        assert os.path.exists(file_path), f'There is no checkpoints file!'
        ckpt = torch.load(file_path, map_location=self.device)
        resume_state_dict = ckpt['model'].state_dict()
        resume_state_dict["encoder.pos_embed"] = resume_state_dict.pop("encoder.my_pos_embed")
        self.model.load_state_dict(resume_state_dict, strict=True)  # load weights

         
    @torch.no_grad()
    def run(self, data):
        # pre processing
        self.model.eval()
        # model
        data = self.transform(data)
        data = data.to(self.device)
        output = self.model(data)
        output = output[0].softmax(dim=0).argmax(dim=0)        
        output = output.cpu().numpy()       
        
        return output

    def transform(self, img):
        img = cv2.resize(img, (self.resize_size, self.resize_size))    
        img = torch.from_numpy(img.transpose(2, 0, 1)) / 255.0
        return img.unsqueeze(0)
        

    def pred_to_rgb(self, pred, color_table):
        pred_rgb = np.zeros_like(pred, dtype=np.uint8)
        pred_rgb = np.repeat(np.expand_dims(pred_rgb[:, :], axis=-1), 3, -1)
        for i in range(len(CLASSES)):
            pred_rgb[pred == i] = np.array(color_table[i])  
               
        return pred_rgb
