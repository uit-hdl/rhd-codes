# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:13:31 2019

@author: bpe043
"""

def sheer_image(img):
    """
    # Function that takes a black and white image, and trims off the white horizontal pixels, leaving only the black numbers
	# i starts at +5 to avoid any of the black "borders" of the box from the questoinnaire that might've snuck into the image
    """

    limit = 255 * img.shape[0]
    start = None
    end = None
    i = 5
    while i < img.shape[1]:
        row = img[:, i]
        if row.sum() < limit and start is None:
            start = i
            
        i += 1
        
    i = img.shape[1]-5
	
    stop = False
    while stop is False:
        
        if i < 0:
            end = 0
            break
        
        row = img[:, i]
        if row.sum() < limit and end is None:
            end = i
            stop = True
        
        i -= 1
        
    return start, end
	
def sheer_image_horizontally(img):
    limit = 255 * img.shape[1]
    start = None
    end = None
    i = 5
    while i < img.shape[0]:
        row = img[i, :]
        if row.sum() < limit and start is None:
            start = i
            
        i += 1
        
    i = img.shape[0]-5
	
    stop = False
    while stop is False:
        
        if i < 0:
            end = 0
            break
        
        row = img[i, :]
        if row.sum() < limit and end is None:
            end = i
            stop = True
        
        i -= 1
        
    return start, end
