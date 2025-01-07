import cv2
import time
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from copy import deepcopy
from dataset.dataset import vehicledata
from model.FCN8s import FCN8s
from model.FCN16s import FCN16s
from model.FCN32s import FCN32s
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
        if self.network_name == 'FCN8s':
            model = FCN8s(num_class= self.num_class)
        elif self.network_name == 'FCN16s':
            model = FCN16s(num_class=self.num_class)
        elif self.network_name == 'FCN32s':
            model = FCN32s(num_class=self.num_class)

        return model.to(self.device)


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
        self.model.eval()
        img = self.transform(data[0])
        s_time= time.time()
        img = img.to(self.device)
        e_time= time.time()
        #print(1/ (e_time-s_time))
        # model
        output = self.model(img)
       
        return output


    def transform(self, img):
        img = cv2.resize(img, (256, 256))    
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        return img.unsqueeze(0)
        

    def pred_to_rgb(self, pred, color_table):
        pred = torch.softmax(pred, dim=0)
        pred = torch.argmax(pred, dim=0)
        pred_ = pred.cpu().numpy()
     
        pred_rgb = np.zeros_like(pred_, dtype=np.uint8)
        pred_rgb = np.repeat(np.expand_dims(pred_rgb[:, :], axis=-1), 3, -1)
        for i in range(len(CLASSES)):
            pred_rgb[pred_ == i] = np.array(color_table[i])  
               
        return pred_rgb
