# -*- coding: utf-8 -*-
"""
Created on Mon May 27 09:28:21 2019

@author: bpe043
"""

import cv2
import numpy as np

# This is to facilitate importing my own helper functions
import sys
sys.path.insert(0, '//homer.uit.no/bpe043/Desktop/Test_Projects/HelperFunctions')

from loadImagesFromFolders import return_image_list_folders

def image_centering(image):
    
    num_rows, num_cols = image.shape[:2]
    
    # Calculations
    ret, thresh = cv2.threshold(image, 127, 255, 0)
    
    M = cv2.moments(thresh)
    
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    
    # Centering
    distance_to_center_X = 0
    distance_to_center_Y = 0
    
    if cX < 50:
        distance_to_center_X = 50 - cX      # Needs to move right / X+
    elif cX > 50:
        distance_to_center_X = -(cX - 50)      # Needs to move left / X-
    else:
        distance_to_center_X = 0
        
    if cY < 50:
        distance_to_center_Y = 50 - cY 
    elif cX > 50:
        distance_to_center_Y = -(cY - 50)
    else:
        distance_to_center_Y = 0
    
    # Translations
    translation_matrix = np.float32([ [1, 0, distance_to_center_X], [0, 1, distance_to_center_Y] ])      # Lucky. Know we need to translate 35 in X+ direction
    img_translated = cv2.warpAffine(image, translation_matrix, (num_cols, num_rows))
    
    return img_translated

root_dir = 'input_centered/'

list_of_images = return_image_list_folders(root_dir)


i = 0
for image in list_of_images:    
    
    img = cv2.imread(image, -1)
    
    # Assuming image is grayscale already
    img = np.expand_dims(img, axis=-1)
     
    # Convert the grayscale image to binary image
    ret, thresh = cv2.threshold(img, 127, 255, 0)
     
    # Calculate moments of binary image
    M = cv2.moments(thresh)
    
    # To avoid ZeroDivisionError - will result in some images that are not centered, but should be relatively small percentage
    if M['m10'] == 0:
        i += 1
        continue
    elif M['m01'] == 0:
        i += 1
        continue
    elif M['m00'] == 0:
        i += 1
        continue

    # Center the image
    image = image_centering(img)
    
    cv2.imwrite(list_of_images[i], image)
    
    i += 1
    print(i)
    
