import cv2
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from torchvision import transforms
from PIL import Image

def adaptive_gamma_correction(img):
    img_rgb = img.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    v = hsv[:,:,2]

    mean_v = np.mean(v)
    std_v = np.std(v)
    im_range = round(abs((mean_v - 2 * std_v) - (mean_v + 2 * std_v)), 1)

    if im_range < 1 / 3.0:
        power_value = -np.log2(std_v + 1e-8)
    else:
        power_value = np.exp((1 - (mean_v + std_v)) / 2)

    k = (v ** power_value) + (1 - v ** power_value) * (mean_v ** power_value)
    k = 1 + (1 if mean_v < 0.5 else 0) * (k - 1)

    v_corrected = (v ** power_value) / k
    v_corrected = np.clip(v_corrected, 0, 1)

    hsv[:, :, 2] = v_corrected
    corrected_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    corrected_rgb = (corrected_rgb * 255).astype(np.uint8)

    return corrected_rgb

def resize_train(image, size = 256):
    resize = transforms.Resize(size)
    return resize(image)

def transform(img):
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float() / 255.0    # when image is into network -> image pixel value is between 0 and 1

    return img

if __name__=='__main__':
    image_path = '/storage/sjpark/vehicle_data/Dataset3/train_image'
    train_dir = sorted(os.listdir(image_path))

    for idx, data in enumerate(train_dir):
        img = os.path.join(image_path, train_dir[10])
        img = Image.open(img)
        img = resize_train(img, (256, 256))
        img = np.array(img, dtype=np.uint8)
        img = adaptive_gamma_correction(img)

        plt.imshow(img)
