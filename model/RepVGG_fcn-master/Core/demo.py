import cv2
import time
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

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
        self.model = self.load_network()
        self.load_weight()


    def load_network(self):
        optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
        g2_map = {l: 2 for l in optional_groupwise_layers}
        g4_map = {l: 4 for l in optional_groupwise_layers}
        if self.network_name == "my":
            from model.RepVGG_fcn8s_sj import RepVGG
            model = RepVGG(num_blocks=[2, 3, 3, 3, 3], num_classes=21, width_multiplier=[1, 1, 1, 1], override_groups_map=None, deploy=True).to(self.device)
            return model
            
        elif self.network_name =="my_m":
            from model.RepVGG_fcn8s_sj_modify import RepVGG
            model = RepVGG(num_blocks=[2, 3, 3, 3, 3], num_classes=21, width_multiplier=[1, 1, 1, 1], override_groups_map=None, deploy=True).to(self.device)                                
              
            return model


    def load_weight(self):
        file_path = self.weight_path
        assert os.path.exists(file_path), f'There is no checkpoints file!'
        print("Loading saved weighted {}".format(file_path))
        ckpt = torch.load(file_path, map_location=self.device)
        from collections import OrderedDict

        if isinstance(ckpt, OrderedDict):
            del ckpt['stage6.1.rbr_reparam.weight']
            del ckpt['stage6.1.rbr_reparam.bias']
            del ckpt['stage6.2.rbr_reparam.weight']
            del ckpt['stage6.2.rbr_reparam.bias']
            #
            try:
                self.model.load_state_dict(ckpt, strict=False)
            except:
                raise  # load weights

         
    @torch.no_grad()
    def run(self, data):
        # pre processing
        self.model.eval()
        img = self.transform(data)
        img = img.to(self.device)
        # model
        output = self.model(img)
        output = torch.softmax(output[0], dim=0)
        output = torch.argmax(output, dim=0).to(torch.int8)
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
