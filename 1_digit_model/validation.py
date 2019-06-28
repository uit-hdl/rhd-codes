# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 08:46:13 2019

@author: bpe043
"""
from tabulate import tabulate
import cv2
import os
import numpy as np
import random

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

import matplotlib.pyplot as plt




# Labeling function - returns training set X with labels y. Optinal copy parameter, if True returns a list of image paths. Used for testing
def read_and_process_image(list_of_images, copy):
    
    X = []  # images
    y = []  # labels
    
    copy_list = list_of_images        
    
    i = 0
    for image in list_of_images :
        X.append(cv2.imread(image, -1))
        
        print(i)
        i = i+1
        
        # Get the labels
        if 'dash' in image  or '-' in image:
            y.append(0)
        elif 'one' in image:
            y.append(1)
        elif 'two' in image:
            y.append(2)
        elif 'three' in image:
            y.append(3)
        elif 'four' in image:
            y.append(4)
        elif 'five' in image:
            y.append(5)
        elif 'six' in image:
            y.append(6)
        elif 'seven' in image:
            y.append(7)
        elif 'eight' in image:
            y.append(8)
        elif 'nine' in image:
            y.append(9)
            
    
    if copy == True:
        return X, y, copy_list        
    
    return X, y

# Function to evaluate the confidence values for image prediction
def evaluate_confidence_scores(confidence_values):
    
    # Find max value, and second highest value. 
    # If there is a large difference (> 15%) between the max and second highest values, return just the max
    # If MAX - Second_Highest < threshold then too close to tell -> return both values
        
    # List of return values. Either [Max index, Max value] or [Max index, Max value, Second Highest index, Second Highest value]
    # If the image cannot be confidently classified, return a 0 to indicate, followed by value and prediction
    return_list = []
    
    #Threshold value, can be tuned
    threshold = 0.15    
    
    # Find the confidence of the max value, and it's index
    max_value = max(confidence_values)
    max_index = np.where(confidence_values == np.amax(confidence_values))[0][0]
    
    # If the best possible value (max) is less than 20%, the image cannot be classified confidently
    if max_value > 0.20:
            
        # Create a copy of the values, and remove the previous max from the copy
        temp_list = confidence_values.copy()
        temp_list = np.delete(temp_list, np.where(temp_list == np.amax(confidence_values)))    
        
        # Find the confidence of the second highest value, and it's index in the ORIGINAL list of confidence values
        second_highest_value = max(temp_list)
        second_highest_index = np.where(confidence_values == second_highest_value)[0][0]
        
        
        if (max_value - second_highest_value > threshold):
            return_list.append([max_index, max_value])
        
        else:
            return_list.append([max_index, max_value, second_highest_index, second_highest_value])
            
        return return_list
    
    else:
        return_list.append([999, max_value, max_index])
        return return_list

# Prediction test
test_dir = 'input/centered_eights'
test_imgs = ['input/centered_eights/{}'.format(i) for i in os.listdir(test_dir)]   # Get test images    

random.shuffle(test_imgs)
    
test_images = 200
X_test, y_test, image_names = read_and_process_image(test_imgs[:test_images], True) #y_test will in this case be empty
x = np.array(X_test)
x = np.expand_dims(x, axis=-1)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load model
model = load_model('Centered_BalancedTraining_BalancedValidation_sparseCategoricalCrossentropy_ESValAcc_SimpleTrainGen_Model.HDF5')
 
# For loop to test our model
i = 0
plt.figure()

headers = ['Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5', 'Label 6', 'Label 7', 'Label 8', 'Label 9']
  
# We know that all the images are supposed to be 8 IN THIS CASE
correct = 0
almost_correct = 0

for batch in test_datagen.flow(x, batch_size=1):
    pred = model.predict(batch)                     # Pred is an array of 10 probability scores for each class (0-9). Softmax output
    #pred = pred.reshape(pred.shape[1])

    text = ''

    for var in pred:
        
        confidence_values = evaluate_confidence_scores(var)
        
        for values in confidence_values:
            
            # Check if image was able to be classified confidently (With a confidence score above 20%)
            if values[0] == 999:
                text = 'This image could not be classified with a confidence of over 20%, achieveing only ' + str( '%.4f' % (values[1] * 100)) + '% confidence as ' + str(values[2])
                break
            
            # Print out the classification and confidence score
            table = tabulate(pred, headers, tablefmt='fancy_grid')
            
            text = 'This image was classified as ' + str(values[0]) + ' with a confidence score of ' + str( '%.4f' % (values[1] * 100)) + '%'
            if values[0] == 8: correct += 1
            
            if len(values) > 2:
                text += '\It was also classified as ' + str(values[2]) + ' with a confidence score of ' + str( '%.4f' % (values[3] * 100)) + '%'
                if values[2] == 8: almost_correct += 1
        
        
    imgplot = plt.imshow(batch[0].squeeze())
    
    plt.title(text)   
    plt.show()
    print(table)
    
    i += 1

    if i % 200 == 0:
        print('The model predicted that the image was an 8 : ' + str(correct) + ' number of times.')
        print('The number of times 8 was the models second choice, was: ' + str(almost_correct) + ' number of times.')
        break
    
    