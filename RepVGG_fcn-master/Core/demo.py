import cv2
import time
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from copy import deepcopy
from dataset.dataset import vehicledata
from model.RepVGG_fcn8 import RepVGG
import torch.nn.functional as F

import torch.autograd.profiler as profiler

CLASSES = [
    'background', 'vehicle', 'bus', 'truck', 'policeCar', 'ambulance', 'schoolBus', 'otherCar',
    'freespace', 'curb', 'safetyZone', 'roadMark', 'whiteLane',
    'yellowLane', 'blueLane', 'constructionGuide', 'trafficDrum',
    'rubberCone', 'trafficSign', 'warningTriangle', 'fence']


class demo:
    def __init__(self, network_name, resize_size, org_size_w, org_size_h, device, weight_path, num_class):
        self.network_name = network_name
        self.resize_size = resize_size
        self.org_size_w = org_size_w
        self.org_size_h = org_size_h
        self.device = device
        self.weight_path = weight_path
        self.num_class = num_class
        self.color_table = self.get_color_table()
        self.model = self.load_network()
        self.load_weight()


    def load_network(self):
        optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
        g2_map = {l: 2 for l in optional_groupwise_layers}
        g4_map = {l: 4 for l in optional_groupwise_layers}
        if self.network_name == "a0":
            model = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=self.num_class,
              width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=True).to(self.device)
              
            return model
        elif self.network_name == "b0":
            model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.num_class,
              width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=True).to(self.device)
              
            return model


    def load_weight(self):
        file_path = self.weight_path
        assert os.path.exists(file_path), f'There is no checkpoints file!'
        print("Loading saved weighted {}".format(file_path))
        ckpt = torch.load(file_path, map_location=self.device)
        from collections import OrderedDict
        if isinstance(ckpt, OrderedDict):
            self.model.load_state_dict(ckpt, strict=True)

        
    def get_color_table(self):
        color_table = {0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0), 3: (0, 0, 128), 4: (128, 128, 0),
                   5: (128, 0, 128), 6: (0, 128, 128), 7: (128, 128, 128), 8: (0, 64, 64),
                   9: (64, 64, 64), 10: (0, 0, 192), 11: (192, 0, 192), 12: (0, 192, 192),
                   13: (192, 192, 192), 14: (64, 128, 0), 15: (192, 0, 128), 16: (64, 128, 128),
                   17: (192, 128, 128), 18: (128, 64, 0), 19: (128, 192, 0), 20: (0, 64, 128)}

        
        return color_table
         
    @torch.no_grad()
    def run(self, data):
        # pre processing
        self.model.eval()
      
        img = self.transform(data)
        s_time = time.time()
        img = img.to(self.device)
        e_time = time.time()
        #print(1 / (e_time - s_time))
        # model
         
        output = self.model(img)
        
       
        output = self.pred_to_rgb(output[0], self.color_table)    
 
        return output


    def transform(self, img):
        img1 = np.array(img, dtype=np.uint8)
        img2 = img1.astype(np.float32)
        img3 = img2.transpose(2, 0, 1)
        img4 = torch.from_numpy(img3).float() / 255.0
        img5 = img4.unsqueeze(0)

        return img5
        

    def pred_to_rgb(self, pred, color_table):
        pred = pred.softmax(dim=0).argmax(dim=0)
       
        pred = pred.to('cpu', non_blocking=True)
       
        pred_rgb = np.zeros_like(pred, dtype=np.uint8)
        pred_rgb = np.repeat(np.expand_dims(pred_rgb[:, :], axis=-1), 3, -1)
         
        for i in range(len(CLASSES)):
            pred_rgb[pred == i] = np.array(color_table[i])
            
        return pred_rgb

