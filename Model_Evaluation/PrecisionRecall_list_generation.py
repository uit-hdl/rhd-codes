# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 09:26:30 2019

@author: bpe043
"""
import cv2
import os

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

import pickle

# This is to facilitate importing my own helper functions
import sys
sys.path.insert(0, '//homer.uit.no/bpe043/Desktop/Test_Projects/HelperFunctions')

from loadImagesFromFolders import return_image_list_labels_folders_3digit



#Load model and create data generator
model = load_model('3_digit/Tresiffer_optimum_weightsVGG16_256_3D_1000.HDF5')
validation_datagen = ImageDataGenerator(rescale=1./255)

#Get validation images
root_dir = '3_digit/Validation set/'

list_of_images, list_of_labels = return_image_list_labels_folders_3digit(root_dir)

# Make a prediction for each image in List of images:
list_of_predictions = []    # Holds the predicted label for each image, used later
list_of_prediction_values = []  # Holds the confidence score of the assigned label. (The MAX confidence score for the prediction)
full_prediction_values = []     # Holds all the confidence scores of the prediction

for image in list_of_images :
    img = cv2.imread(image, -1)
    
     # Prediction
    for batch in validation_datagen.flow(img, batch_size=1):
        pred = model.predict(batch)
        
        predictions = pred.reshape(pred.shape[1])
        full_prediction_values.append(predictions)
        
        # Find the confidence of the max value, and it's index
        max_value = max(predictions)
        max_index = np.where(predictions == np.amax(predictions))[0][0]
        list_of_predictions.append(max_index)
        list_of_prediction_values.append(max_value)
        
        break       # Exit after the prediction has been made, and move to the next image
        
# Recall and Precision with bonus F1 score
with open('3_digit/labels.txt', 'wb') as fp:
    pickle.dump(list_of_labels, fp)

with open('3_digit/predictions.txt', 'wb') as fp:
    pickle.dump(list_of_predictions, fp)
    
with open('3_digit/prediction_values.txt', 'wb') as fp:
    pickle.dump(list_of_prediction_values, fp)
    
with open('3_digit/full_prediction_values.txt', 'wb') as fp:
    pickle.dump(full_prediction_values, fp)