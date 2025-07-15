import torch
import cv2
import numpy as np



def histogram_equal(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    l, a, b = cv2.split(img)

    lab_clahe = cv2.merge((cv2.equalizeHist(l), a, b))
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_YCrCb2RGB)

    return result

def clahe(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    l, a, b = cv2.split(img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_YCrCb2RGB)

    return result

def retinex_MSRCR(img, sigma_list=[5, 15, 30], gain=1.0, offset=0):
    # Dataset : sigma_list = [5, 15, 30], gain=1.0, offset=0
    # Dataset3 : sigma_list = [15, 80, 250], gain=1.0, offset=0
    img = img.astype(np.float32) + 1.0
    log_R = np.zeros_like(img)

    for sigma in sigma_list:
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        log_R += np.log(img) - np.log(blur + 1.0)

    log_R /= len(sigma_list)

    sum_channels = np.sum(img, axis=2, keepdims=True)
    crf = np.log(img / (sum_channels + 1e-6) + 1.0)


    msrcr = gain * log_R * crf + offset

    msrcr = np.clip(msrcr, 0, None)
    msrcr = cv2.normalize(msrcr, None, 0, 255, cv2.NORM_MINMAX)
    msrcr = np.clip(msrcr, 0, 255).astype(np.uint8)

    return msrcr

def retinex_MSR(img, sigma_list= [5, 15, 30]):
    # Dataset : sigma_list = [5, 15, 30]
    # Dataset3 : sigma_list = [15, 80, 250]
    img = img.astype(np.float32) + 1.0
    result = np.zeros_like(img)
    for sigma in sigma_list:
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        result += np.log(img) - np.log(blur + 1)
    result = result / len(sigma_list)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(result)


def gammacorrection(img, gamma=2.0):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)
    
    
    

