#! /usr/bin/env python3

import sys
import rospy
import numpy as np
from sensor_msgs.msg import Image

def imgmsg_to_cv2(img_msg):
    if img_msg.encoding != "bgr8":
        rospy.logger("Change img_msg encoding to bgr8")
    dtype = np.dtype("uint8")
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    imge_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), dtype=dtype, buffer=img_msg.data)

    if img_msg.is_bigendian == (sys.byteorder=='little'):
        image_opencv = image_opencv.byteswap().newbyteorder()

    return image_opencv
