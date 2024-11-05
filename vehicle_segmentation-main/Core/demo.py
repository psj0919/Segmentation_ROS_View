import cv2
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from dataset.dataset import vehicledata
from model.FCN8s import FCN8s
from model.FCN16s import FCN16s
from model.FCN32s import FCN32s
import torch.nn.functional as F

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

        return model


    def load_weight(self):
        file_path = self.weight_path
        assert os.path.exists(file_path), f'There is no checkpoints file!'
        print("Loading saved weighted {}".format(file_path))
        ckpt = torch.load(file_path, map_location=self.device)
        resume_state_dict = ckpt['model'].state_dict()

        self.model.load_state_dict(resume_state_dict, strict=True)  # load weights


    def run(self, data):
        # pre processing
        img = self.transform(data)
        img = self.resize_train(img, self.resize_size)
        img = img.unsqueeze(0)
        img_ =self.matplotlib_imshow(img[0])


        # model
        output = self.model(img)

        # make_segmentation
        segmentation_iamge = self.pred_to_rgb(output[0])
        segmentation_iamge = self.resize_output(segmentation_iamge, self.org_size_w, self.org_size_h)
        segmentation_iamge = cv2.addWeighted(data, 1, segmentation_iamge, 0.5, 0)

        return segmentation_iamge


    def transform(self, img):
        img = img.astype(np.float64)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float() / 255.0    # when image is into network -> image pixel value is between 0 and 1

        return img
        
    def resize_train(self, image, size):
        return F.interpolate(image.unsqueeze(0), size, mode='bilinear', align_corners=True).squeeze(0)

    def resize_output(self, image, w_size, h_size):
    	output = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    	output = F.interpolate(output, size=(w_size, h_size), mode='bilinear', align_corners=True).squeeze(0)
    	output = self.matplotlib_imshow(output)
    	return output 
        
        
    def matplotlib_imshow(self, img):
        assert len(img.shape) == 3
        npimg = img.numpy()
        return (np.transpose(npimg, (1, 2, 0))[:, :, ::-1] * 255).astype(np.uint8)

    def pred_to_rgb(self, pred):
        assert len(pred.shape) == 3
        #
        threshold = 0.5
        pred = pred.softmax(dim=0)
        pred, class_index = pred.max(dim=0)
        pred = torch.where(pred > threshold, class_index, 0)
        #
        pred = pred.detach().cpu().numpy()
        #
        color_segmentation_image = np.zeros((self.resize_size, self.resize_size, 3))
        #
        color_table = {0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0), 3: (0, 0, 128), 4: (128, 128, 0),
                       5: (128, 0, 128), 6: (0, 128, 128), 7: (128, 128, 128), 8: (0, 64, 64),
                       9: (64, 64, 64), 10: (0, 0, 192), 11: (192, 0, 192), 12: (0, 192, 192),
                       13: (192, 192, 192), 14: (64, 128, 0), 15: (192, 0, 128), 16: (64, 128, 128),
                       17: (192, 128, 128), 18: (128, 64, 0), 19: (128, 192, 0), 20: (0, 64, 128)}
        #
        for i in range(len(CLASSES)):
            color_segmentation_image[pred == i] = np.array(color_table[i])

        return color_segmentation_image

#-------------------------------detected_class.py-------------------------------#
# #! /usr/bin/env python3
# import cv2
# import torch
# # import Bridge
# # import rospy
# import numpy as np
# # import sensor_msgs.msg import Image
# # from std_msg.msg import Int32
# # from vehicle_segmentation.Core import demo.py
#
#
# Resize_size = 256
# Weight_path = '/storage/sjpark/vehicle_data/checkpoints/FCN8/256/fcn_epochs:200_optimizer:adam_lr:0.0001_modelfcn8.pth'
# GPU_ID = '0'
# DEVICE = torch.device('cuda:0')
# network_name = 'FCN8s' # FCN16s | FCN32s
# num_class = 20
#
# Detected_class = demo(network_name, Resize_size, DEVICE, Weight_path, num_class)
#
#
# class detected_class:
#     def __init__(self):
#         self.input_image = rospy.Subscriber("/fps_controller/image_raw", Image, self.callback)
#         self.result_image = rospy.Publisher("/detected_class/result_image", Image, queue_size=10)
#         self.bridge = CvBridge()
#
#     def callback(self, data):
#         try:
#             cv_input_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
#             result_image = demo.run(cv_input_image)
#             self.result_image.publish(result_image)
#         except:
#             print("Error")
#
#
# def main():
#     class_detector = detected_class()
#     rospy.init_node('class_detector', anonymous=True)
#
#     try:
#         rospy.spin()
#     except KeyboardInterrupt:
#         print("shut down")
#         cv2.destroyWindow()
#
#---------------------------------------------------------------------------------------#

# def transform(img):
#     img = img[:, :, ::-1]  # RGB -> BGR
#     img = img.astype(np.float64)
#     img = img.transpose(2, 0, 1)
#     img = torch.from_numpy(img).float() / 255.0    # when image is into network -> image pixel value is between 0 and 1
#
#     return img
#
# def resize_train(image, size):
#     return F.interpolate(image.unsqueeze(0), size, mode='bilinear', align_corners=True).squeeze(0)
#
#
# def matplotlib_imshow(img):
#     assert len(img.shape) == 3
#     npimg = img.numpy()
#     return (np.transpose(npimg, (1, 2, 0))[:, :, ::-1] * 255).astype(np.uint8)
#
# def pred_to_rgb(pred):
#     assert len(pred.shape) == 3
#     #
#     pred = pred.softmax(dim=0).argmax(dim=0).to('cpu')
#     #
#     pred = pred.detach().cpu().numpy()
#     #
#     color_segmentation_image = np.zeros((512, 512, 3))
#     #
#     color_table = {0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0), 3: (0, 0, 128), 4: (128, 128, 0),
#                    5: (128, 0, 128), 6: (0, 128, 128), 7: (128, 128, 128), 8: (0, 64, 64),
#                    9: (64, 64, 64), 10: (0, 0, 192), 11: (192, 0, 192), 12: (0, 192, 192),
#                    13: (192, 192, 192), 14: (64, 128, 0), 15: (192, 0, 128), 16: (64, 128, 128),
#                    17: (192, 128, 128), 18: (128, 64, 0), 19: (128, 192, 0), 20: (0, 64, 128)}
#     #
#     for i in range(len(CLASSES)):
#         color_segmentation_image[pred == i] = np.array(color_table[i])
#
#     return color_segmentation_image
#
#
# def load_weight():
#     file_path = '/storage/sjpark/vehicle_data/checkpoints/FCN8/512/fcn_epochs:200_optimizer:adam_lr:0.0001_modelfcn8.pth'
#     assert os.path.exists(file_path), f'There is no checkpoints file!'
#     print("Loading saved weighted {}".format(file_path))
#     ckpt = torch.load(file_path)
#     resume_state_dict = ckpt['model'].state_dict()
#
#     model.load_state_dict(resume_state_dict, strict=True)
#
# if __name__ == '__main__':
#
#     img = '/storage/sjpark/vehicle_data/Dataset/test_image/88_081217_221008_24.jpg'
#     img = cv2.imread(img)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = transform(img)
#     img = resize_train(img, 512)
#     img = img.unsqueeze(0)
#
#     from model.FCN8s import FCN8s
#     model = FCN8s(num_class=21)
#     load_weight()
#
#     output = model(img)
#
#     img = matplotlib_imshow(img[0])
#     seg_img = pred_to_rgb(output[0])
#
#     plt.imshow(img, cmap="gray")
#     plt.imshow(seg_img)
#     plt.axis("off")
#     plt.show()



