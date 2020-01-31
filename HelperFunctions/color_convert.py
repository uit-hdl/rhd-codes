# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:13:31 2019

@author: bpe043
"""

import cv2
import numpy as np

def convert_img_gray(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # lower mask
    lower_red = np.array([0, 45, 50])
    upper_red = np.array([20, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    
    # upper mask
    lower_red = np.array([160, 50, 50])
    upper_red = np.array([190, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    
    mask = mask0 + mask1
    
    # Check for any red pixels (Means there exists writing in the image)        
    red_pix = np.count_nonzero(mask)
    
    # If the image contains red pixels, return a tuple of True and the converted image
    if red_pix > 300:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return (True, gray)
    # If the image contains no red pixels, return a tuple of False and the oiriginal image
    else:
        return (False, img) 
    
def convert_img_bw(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask
    lower_red = np.array([0, 45, 50])
    upper_red = np.array([20, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask
    lower_red = np.array([160, 50, 50])
    upper_red = np.array([190, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    mask = mask0 + mask1

    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask == 0)] = 0
    b_w = cv2.split(output_hsv)[2]
    retval, b_w = cv2.threshold(b_w, 100, 255, cv2.THRESH_BINARY)
    b_w = cv2.GaussianBlur(b_w, (3, 3), 0)
    b_w = cv2.bitwise_not(b_w)

    return b_w