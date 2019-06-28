# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 15:56:34 2019

@author: bpe043
"""

import os
import cv2
import pickle


# Function to read image files as image arrays, and label them correctly according to birthplace
# Input is the root directory where subdirectories with images lie
def read_and_label_birthplaces(root):
    
    X = [] # Image list
    Y = [] # Label list
    
    # Get all the subdirectories (that contain the images) from root
    places = [x[0] for x in os.walk(root)]
    del places[0]   # Remove the first element, as it's just root
    
    i = 0
    l = 0
    # Iterate over each subfolder
    for place in places: 
        for (path, _, filenames) in os.walk(place):    #Get the path and a list of all the files in the subfolder   
            for name in filenames:
                x = cv2.imread(path + '/' + name, 1)    # Concatenate the path with the filename to get input to cv2.imread in the correct format. ex. Birthplaces/Bergen/name
                x = cv2.resize(x, (280, 210))
                X.append(x)
                print('Image number {} is done!'.format(i))
                i += 1
        #label = place.split('/')[1]                 # Get the place name from the place string (i.e get 'Bergen' from 'Birthplaces/Bergen')
        #Y.extend( [label for i in range(1000)] )    # Since we are grabbing 1000 images, make 1000 labels
        Y.extend( [l for x in range(1000)] )
        l += 1
        print('One class done')
        
    return X, Y
    


root = 'Birthplaces/'

image_list, label_list = read_and_label_birthplaces(root)

with open('birthplace_images.pkl', 'wb') as f:
    pickle.dump(image_list, f)
    

with open('birthplace_labels.pkl', 'wb') as f:
    pickle.dump(label_list, f)