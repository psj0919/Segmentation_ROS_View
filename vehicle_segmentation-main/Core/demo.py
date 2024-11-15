import cv2
import time
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from dataset.dataset import vehicledata
from model.FCN8s import FCN8s
from model.FCN16s import FCN16s
from model.FCN32s import FCN32s
import torch.nn.functional as F
torch.set_num_threads(8)
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


    def run(self, data):
        # pre processing
        img = np.array(data, dtype=np.uint8)
        img = self.transform(img)
        img = self.resize_train(img, self.resize_size)
        img = img.unsqueeze(0)
        
        


        # model
        self.model.eval()
        s_time = time.time()   
        output = self.model(img.to(self.device))
        e_time = time.time()
        #print(1 / (e_time - s_time))

        # make_segmentation

        s_time = time.time()           
        segmentation_iamge = self.pred_to_rgb(output[0])
        e_time = time.time()
        #print("pred_to_rgb:", 1 / (e_time - s_time))
                
        segmentation_iamge = self.resize_output(segmentation_iamge, self.org_size_w, self.org_size_h)
        segmentation_iamge = cv2.addWeighted(data, 1, segmentation_iamge, 0.5, 0)
      
        return segmentation_iamge


    def transform(self, img):
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float() / 255.0    # when image is into network -> image pixel value is between 0 and 1

        return img
        
    def resize_train(self, image, size):
        return F.interpolate(image.unsqueeze(0), size, mode='bilinear', align_corners=True).squeeze(0)

    def resize_output(self, image, w_size, h_size):
    	#output = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
    	output = image.permute(2, 0, 1).unsqueeze(0)
    	output = F.interpolate(output, size=(w_size, h_size), mode='bilinear', align_corners=True).squeeze(0)
    	#output = self.matplotlib_outputshow(output)
    	output = output.permute(1, 2, 0)
    	return output.detach().cpu().numpy().astype(np.uint8)
        
        
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
        pred = pred.softmax(dim=0)
        pred = pred.argmax(dim=0)
        #pred = pred.detach().cpu().numpy()


        #
        w, h = pred.shape[0], pred.shape[1]
        pred_rgb = np.zeros((w, h), dtype=np.uint8)
        pred_rgb = np.repeat(np.expand_dims(pred_rgb[:,:],axis=-1),3, -1)
        #
        color_table = {0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0), 3: (0, 0, 128), 4: (128, 128, 0),
                       5: (128, 0, 128), 6: (0, 128, 128), 7: (128, 128, 128), 8: (0, 64, 64),
                       9: (64, 64, 64), 10: (0, 0, 192), 11: (192, 0, 192), 12: (0, 192, 192),
                       13: (192, 192, 192), 14: (64, 128, 0), 15: (192, 0, 128), 16: (64, 128, 128),
                       17: (192, 128, 128), 18: (128, 64, 0), 19: (128, 192, 0), 20: (0, 64, 128)}
                       
        #
        color_table_tensor = torch.tensor(list(color_table.values()), device = self.device, dtype=torch.float)
        pred_rgb = torch.tensor(pred_rgb).type('torch.FloatTensor')

        #s_time2 = time.time()
        #for i in range(len(CLASSES)):
            #pred_rgb[pred == i] = torch.tensor(np.array(color_table[i])).type('torch.FloatTensor')
        pred_rgb = color_table_tensor[pred]   
        #e_time2 = time.time()
        #print("segment_view = {}".format(1 / (e_time2 - s_time2)))
        

        return pred_rgb
        

        



