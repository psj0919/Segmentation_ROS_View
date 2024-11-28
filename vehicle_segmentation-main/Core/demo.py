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

color_table = {0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0), 3: (0, 0, 128), 4: (128, 128, 0),
               5: (128, 0, 128), 6: (0, 128, 128), 7: (128, 128, 128), 8: (0, 64, 64),
               9: (64, 64, 64), 10: (0, 0, 192), 11: (192, 0, 192), 12: (0, 192, 192),
               13: (192, 192, 192), 14: (64, 128, 0), 15: (192, 0, 128), 16: (64, 128, 128),
               17: (192, 128, 128), 18: (128, 64, 0), 19: (128, 192, 0), 20: (0, 64, 128)}


class demo:
    def __init__(self, network_name, resize_size, org_size_w, org_size_h, device, weight_path, num_class):
        self.network_name = network_name
        self.resize_size = resize_size
        self.org_size_w = org_size_w
        self.org_size_h = org_size_h
        self.device = device
        self.weight_path = weight_path
        self.num_class = num_class
        self.color_table = self.get_color_table(color_table)
        self.model = self.load_network()
        self.load_weight()


    def load_network(self):
        if self.network_name == 'FCN8s':
            model = FCN8s(num_class= self.num_class)
        elif self.network_name == 'FCN16s':
            model = FCN16s(num_class=self.num_class)
        elif self.network_name == 'FCN32s':
            model = FCN32s(num_class=self.num_class)

        return model.to(self.device).eval()


    def load_weight(self):
        file_path = self.weight_path
        assert os.path.exists(file_path), f'There is no checkpoints file!'
        print("Loading saved weighted {}".format(file_path))
        ckpt = torch.load(file_path, map_location=self.device)
        resume_state_dict = ckpt['model'].state_dict()

        self.model.load_state_dict(resume_state_dict, strict=True)  # load weights
        
    def get_color_table(self, table):
        color_table_tensor = torch.tensor(list(table.values()), device = self.device, dtype=torch.int)

        return color_table_tensor
        
    @torch.no_grad()
    def run(self, data):
        # pre processing
        self.model.eval()

        img = self.transform(data)
        # model
        output = self.model(img)    

        return output


    def transform(self, img):
        img1 = np.array(img, dtype=np.uint8)
        img2 = img1.astype(np.float32)
        img3 = img2.transpose(2, 0, 1)
        img4 = torch.from_numpy(img3).float() / 255.0    # when image is into network -> image pixel value is between 0 and 1
        img5 = img4.unsqueeze(0).to(self.device)

        return img5
        
    def resize_train(self, image, size):
        return F.interpolate(image.unsqueeze(0), size, mode='bilinear', align_corners=True).squeeze(0)

        
    def matplotlib_imshow(self, img):
        assert len(img.shape) == 3
        npimg = img.numpy()
        return (np.transpose(npimg, (1, 2, 0))[:, :, ::-1] * 255).astype(np.uint8)


    def matplotlib_outputshow(self, img):
        assert len(img.shape) == 3
        npimg = img.numpy()
        return (np.transpose(npimg, (1, 2, 0))[:, :, ::-1]).astype(np.uint8)

    def pred_to_rgb(self, pred):
        assert len(pred.shape) == 3
        #
        pred = pred.softmax(dim=0).argmax(dim=0)
        #
        pred_rgb = self.color_table[pred.long()]
        return pred_rgb
        

        



