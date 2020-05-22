# -*- coding: utf-8 -*-

""" Version 2.0 """

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.models import load_model
import numpy as np

import pandas as pd
from io import StringIO

import cv2
import csv

import sys
sys.path.insert(0, '//homer.uit.no/bpe043/Desktop/Test_Projects/HelperFunctions')
import db_image_decode as decode_image

from GND_Clustering import ImageSegClustering as splitter


# Normally, the ground truth data will have already been created. But if it needs to be made in the moment, this can be used
def data_split(data, three_digits = False):
    X = []
    y = []
    
    for d in data:
        img = decode_image(d[0])
        
        if len(img.shape) > 2:
            continue
        
        if three_digits == True:
            width = 200
            height = 115
            dim = (width, height)
            
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        
        X.append(img)
        
        y.append(d[1])
        
        
    zero_indexes = [i for i in y if i == '0']
    del y[: len(zero_indexes)]
    del X[: len(zero_indexes)]
    
    X = np.array(X)
    X = np.expand_dims(X, axis=-1)
    y = np.array(y)
    
    return X, y

def one_digit(images, labels, model):
        # When a "validation set" has to be made on the fly. Should not have a need to use this method anymore. Use ground truth arrays instead
# =============================================================================  
#     # Connect to the 1-digit training set and get the data
#     singles_db_path = '\\\\129.242.140.132\\remote\\UtklippsDatabaser\\full_1digit_trainingset.db'
#     db = DbHandler(singles_db_path)
#     data = db.select_all_training_images('split_grey')
#     
#     X, y = data_split(data, three_digit)
#     
#     # Split the data into a 20% validation set
#     X_train_unused, x_true, y_train_unused, y_true = train_test_split(X, y, test_size = 0.20, stratify=y, shuffle = True)
#     
# =============================================================================
    #target names
    target_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']
    
    # Need ground truth images
    x_true = images

    # Need ground truth labels
    y_true = labels
    
    print('Model starting')
    
    # Run prediction on the model with the ground truth images, to get the predicted labels
    y_pred = model.predict(x_true)
    
    print('Model completed')

    # Keep only the most likely predictions
    y_pred2 = np.argmax(y_pred, axis = 1)
    
    # Classification report
    #print(classification_report(y_true.astype(int), y_pred2, target_names = target_names))
    report = classification_report(y_true.astype(int), y_pred2, target_names = target_names, output_dict = True)
    
    df = pd.DataFrame(report).transpose()

    df.to_csv('Validation\\1_digit_report_only_augmented_training.csv', sep=';')
    
    # Save the confidence scores to csv file
# =============================================================================
#     y_pred = np.delete(y_pred, 0, 1)
#     y_pred = y_pred.astype(str)
#     y_pred = np.insert(y_pred, 0, target_names, axis = 0)
# =============================================================================
    
    conf = pd.DataFrame(data = y_pred[0:, 0:],
                    index = range(0, len(y_pred)),
                    columns = y_pred[0, 0:])

    conf.to_csv('Validation\\Confidence_scores_1_digit_only_augmented_training.csv', sep=';', index = False)
    
    print(df)


def three_digit():
        # When a "validation set" has to be made on the fly. Should not have a need to use this method anymore. Use ground truth arrays instead
# =============================================================================
#     # Connect to the 3-digit training set, and get the data
#     db_path = '\\\\129.242.140.132\\remote\\UtklippsDatabaser\\full_3digit_trainingset.db'
#     db = DbHandler(db_path)
#     data = db.select_all_training_images_3digit('greyscale', 'cells')
#     
#     X, y = data_split(data, three_digit)
#     
#     # 3-digit, classes with too few members to do stratification
#     X_train_unused, x_true, y_train_unused, y_true = train_test_split(X, y, test_size = 0.20, shuffle = True)
# =============================================================================

    # Need ground truth images
    x_true = np.load('C:\\Models\\Ground_truth_arrays\\3_digit_greyscale_ground_truth_images.npy')
    
    # Need ground truth labels
    y_true = np.load('C:\\Models\\Ground_truth_arrays\\3_digit_greyscale_ground_truth_labels.npy')
    
    # Mapping for the labels
    mapping = np.load('C:\\Models\\Ground_truth_arrays\\3_digit_ground_truth_mapping.npy')

    model = load_model('C:\\Models\\Stratified_model_3-digit_greyscale_fold_9.h5')

    # Run prediction on the model with the ground truth images, to get the predicted labels
    y_pred = model.predict(x_true)

    y_pred2 = np.argmax(y_pred, axis = 1)

    # Get the report
    report = classification_report(y_true, y_pred2, output_dict = True)
    
    df = pd.DataFrame(report).transpose()
    
    # Clean up some data that gets messed up when converting to a dataframe
    df['precision'].iloc[-3] = np.nan
    df['recall'].iloc[-3] = np.nan
    df['support'].iloc[-3] = df['support'].iloc[-2] 

    # Map the classes back to actual occupation codes
    index = df.index[:-3].astype(int).to_list()
    new_index = []
    for i in index:
        code = mapping[i]
        new_index.append(code)
    
    new_index.append('accuracy')
    new_index.append('macro avg')
    new_index.append('weighted avg')
    #df.index = new_index
    
    #columns =[' ', 'precision', 'recall', 'f1-score', 'support']
    #df.columns = columns
    df['class'] = new_index
    columns =['class', 'precision', 'recall', 'f1-score', 'support']
    df = df[columns]
    
    print(df)

    # Getting confidence scores to csv
    conf = pd.DataFrame(data = y_pred[0:, :-1],
                        index = range(0, len(y_pred)),
                        columns = ['Class ' + c for c in [str(mapping[x]) for x in range(0, 264)] ])
    
    
    conf.to_csv('Validation\\Confidence_scores_3_digit.csv', sep = ';', index=False)
                        #columns = [x for x in range(0, 265)])

    df.to_csv('Validation\\3_digit_report_2.csv', sep = ';', index=False)


def predict_and_report(model, x_arr, y_arr, digits, mapping = None, target_names = None):
    
    if digits == '3':
            # Run prediction on the model with the ground truth images, to get the predicted labels
        y_pred = model.predict(x_arr)
    
        y_pred2 = np.argmax(y_pred, axis = 1)
    
        # Get the report
        report = classification_report(y_arr, y_pred2, output_dict = True)
        
        df = pd.DataFrame(report).transpose()
        
        # Clean up some data that gets messed up when converting to a dataframe
        df['precision'].iloc[-3] = np.nan
        df['recall'].iloc[-3] = np.nan
        df['support'].iloc[-3] = df['support'].iloc[-2] 
    
        # Map the classes back to actual occupation codes
        index = df.index[:-3].astype(int).to_list()
        new_index = []
        for i in index:
            code = mapping[i]
            new_index.append(code)
        
        new_index.append('accuracy')
        new_index.append('macro avg')
        new_index.append('weighted avg')
        
        df['class'] = new_index
        columns =['class', 'precision', 'recall', 'f1-score', 'support']
        df = df[columns]
    
        # Getting confidence scores
        conf = pd.DataFrame(data = y_pred[0:, :-1],
                            index = range(0, len(y_pred)),
                            columns = ['Class ' + c for c in [str(mapping[x]) for x in range(0, 264)] ])
        
        # Can either return the predictions and confidence scrores, or save them as csv in the function
        # Returning for now
        
        return (df, conf)
        #conf.to_csv('Validation\\Confidence_scores_3_digit.csv', sep = ';', index=False)
        #df.to_csv('Validation\\3_digit_report_2.csv', sep = ';', index=False)
        
    else:
# =============================================================================
#         # Run prediction on the model with the ground truth images, to get the predicted labels
#         y_pred = model.predict(x_arr)
#     
#         # Keep only the most likely predictions
#         y_pred2 = np.argmax(y_pred, axis = 1)
#         
#         # Classification report
#         #print(classification_report(y_true.astype(int), y_pred2, target_names = target_names))
#         report = classification_report(y_arr.astype(int), y_pred2, target_names = target_names, output_dict = True)
#         
#         df = pd.DataFrame(report).transpose()
# =============================================================================
        #target names
        target_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']
        
        # Run prediction on the model with the ground truth images, to get the predicted labels
        
        # Confidence scores
        y_pred = model.predict(x_arr)
        
        # Keep only the most likely predictions
        # Labels
        y_pred2 = np.argmax(y_pred, axis = 1)
        
        # Classification report
        #print(classification_report(y_true.astype(int), y_pred2, target_names = target_names))
        report = classification_report(y_arr.astype(int), y_pred2, target_names = target_names, output_dict = True)
        
        df = pd.DataFrame(report).transpose()
        
        
        # Save the confidence scores to csv file
        #y_pred = np.delete(y_pred, 0, 1)
        #y_pred = y_pred.astype(str)
        #y_pred = np.insert(y_pred, 0, target_names, axis = 0)
        #y_pred = np.delete(y_pred, 0, 1)        # Remove the first column of confidence scores for 0. It shouldn't have been included in the first place
        
        conf = pd.DataFrame(data = y_pred[0:, 0:],
                        index = range(0, len(y_pred)),
                        columns = y_pred[0, 0:])
        
        conf.columns = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']
    
    
        return (df, conf)
        #df.to_csv('Validation\\1_digit_report_augmented.csv', sep=';')
        #conf.to_csv('Validation\\Confidence_scores_1_digit_augmented.csv', sep=';', index = False)
        
        
        

def both_models():
    
    # Need ground truth images
    x_3digit_truth = np.load('C:\\Models\\Ground_truth_arrays\\3_digit_original_ground_truth_images.npy')
    
    # Need ground truth labels
    y_3digit_truth = np.load('C:\\Models\\Ground_truth_arrays\\3_digit_original_ground_truth_labels.npy')
    
    # Mapping for the labels
    mapping = np.load('C:\\Models\\Ground_truth_arrays\\3_digit_original_ground_truth_mapping.npy')
    
    labels_3digit = [mapping[x] for x in y_3digit_truth]
    target_names_1digit = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']
    
    # Lists to hold the images (and their labels) that could not be split
    x_not_splittable = []
    y_not_splittable = []
    
    # Lists to hold the 3 digit and 1 digit version (along with their respective labels) of the iomages
    x_3digit = []
    x_1digit = []
    
    y_3digit = []
    y_1digit = []

    index = 0
    while index < len(x_3digit_truth):
        img = x_3digit_truth[index]
        label = y_3digit_truth[index]
        
        # Labels of 0 are not valid labels
        if '0' in labels_3digit[index]:
            print('Skipping label {} because it contains a 0'.format(labels_3digit[index]))
            index += 1
            continue
        
        #unique labels, find 0 labels. labels at least Containing a 0
        #Se også hvor mange bilder som detter ut
        
        # If the image can be splt into into 1 digits, we add both 3 and 1 digit images to their own lists
        # If it can't we keep the image in another list
        split = splitter.split_and_convert(img)
        
        if split is None:
            index += 1
            x_not_splittable.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            y_not_splittable.append(label)
            
        else:
            # We put the splittable 3 digit image into the list, same with the label
            x_3digit.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            y_3digit.append(label)
            
            # If any of the split images, for whatever reason, didn't get properly converted, we redo it
            if len(split[2][0].shape) > 2:
                split[2][0] = cv2.cvtColor(split[2][0], cv2.COLOR_BGR2GRAY)
                
            if len(split[2][1].shape) > 2:
                split[2][1] = cv2.cvtColor(split[2][1], cv2.COLOR_BGR2GRAY)
                
            if len(split[2][2].shape) > 2:
                split[2][2] = cv2.cvtColor(split[2][2], cv2.COLOR_BGR2GRAY)

             # Then we put the split image into it's list, along with Their leabels
            x_1digit.append(split[2][0])
            x_1digit.append(split[2][1])
            x_1digit.append(split[2][2])
            
            # Need to get the actuyal digit instead of the model label for this step
            label = str(labels_3digit[index])
            y_1digit.append(int(label[0]))
            y_1digit.append(int(label[1]))
            y_1digit.append(int(label[2]))
            
            index += 1
        
    # Load in our models
    model_3digit = load_model('C:\\Models\\Stratified_model_3-digit_greyscale_fold_9.h5')
    model_1digit = load_model('C:\\Models\\Stratified_model_1-digit_grey_fold_9.h5')
    
    # Turn the lists into arrays, and give the X arrays an additional dimension for use in the model
    x_not_splittable = np.array(x_not_splittable)
    x_3digit = np.array(x_3digit)
    x_1digit = np.array(x_1digit)
    
    x_not_splittable = np.expand_dims(x_not_splittable, axis=-1)
    x_3digit = np.expand_dims(x_3digit, axis=-1)
    x_1digit = np.expand_dims(x_1digit, axis=-1)
    
    y_not_splittable = np.array(y_not_splittable)
    y_3digit = np.array(y_3digit)
    y_1digit = np.array(y_1digit)
    
    # Now we can send the arrays into model.predict()
    df_unsplittable, conf_unsplittable = predict_and_report(model_3digit, x_not_splittable, y_not_splittable, '3', mapping = mapping)
    df_3digit, conf_3digit = predict_and_report(model_3digit, x_3digit, y_3digit, '3', mapping = mapping)
    df_1digit, conf_1digit = predict_and_report(model_1digit, x_1digit, y_1digit, '1', target_names = target_names_1digit)
    
    # Save results
    df_unsplittable.to_csv('Validation\\3digit_unsplittable_results.csv', sep=';')
    conf_unsplittable.to_csv('Validation\\3digit_unsplittable_confidence_scores.csv', sep=';', index = False)
    
    df_3digit.to_csv('Validation\\3digit_results.csv', sep=';')
    conf_3digit.to_csv('Validation\\3digit_confidence_scores.csv', sep=';', index = False)
    
    df_1digit.to_csv('Validation\\1digit_results.csv', sep=';')
    conf_1digit.to_csv('Validation\\1digit_confidence_scores.csv', sep=';', index = False)
    
    print('Finished! Yuhuu!')
    




both_models()
    
    
# Need ground truth images
#x_true = np.load('C:\\Models\\Ground_truth_arrays\\1_digit_grey_ground_truth_images_only_augmented_training.npy')

# Need ground truth labels
#y_true = np.load('C:\\Models\\Ground_truth_arrays\\1_digit_grey_ground_truth_labels_only_augmented_training.npy')

# Load in a model to evaluate
#model = load_model('C:\\Models\\current models\\Stratified_only_augmented_training_model_1-digit_grey_fold_9.h5')    

#one_digit(x_true, y_true, model)




""" Version 1.0 below """


# =============================================================================
# # -*- coding: utf-8 -*-
# """
# Created on Tue Jun  4 08:46:13 2019
# 
# @author: bpe043
# """
# 
# from tabulate import tabulate
# import cv2
# import os
# import numpy as np
# import random
# 
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import load_model
# 
# import matplotlib.pyplot as plt
# 
# import sqlite3
# 
# import sys
# sys.path.insert(0, '//homer.uit.no/bpe043/Desktop/Test_Projects/HelperFunctions')
# 
# from Database.dbHandler import DbHandler
# from db_image_decode import decode_image
# 
# 
# # Labeling function - returns training set X with labels y. Optinal copy parameter, if True returns a list of image paths. Used for testing
# def read_and_process_image(list_of_images, copy):
#     
#     X = []  # images
#     y = []  # labels
#     
#     copy_list = list_of_images        
#     
#     i = 0
#     for image in list_of_images :
#         print(i)
#         i = i+1
#         
#         img = cv2.imread(image, -1)
#         
#         if img is not None:
#             X.append(cv2.imread(image, -1))
#         else:
#             continue
#         
# 
#         
#         
#         # Get the labels
#         if 'dash' in image  or '-' in image:
#             y.append(0)
#         elif 'one' in image:
#             y.append(1)
#         elif 'two' in image:
#             y.append(2)
#         elif 'three' in image:
#             y.append(3)
#         elif 'four' in image:
#             y.append(4)
#         elif 'five' in image:
#             y.append(5)
#         elif 'six' in image:
#             y.append(6)
#         elif 'seven' in image:
#             y.append(7)
#         elif 'eight' in image:
#             y.append(8)
#         elif 'nine' in image:
#             y.append(9)
#             
#     
#     if copy == True:
#         return X, y, copy_list        
#     
#     return X, y
# 
# # Function to evaluate the confidence values for image prediction
# def evaluate_confidence_scores(confidence_values):
#     
#     # Find max value, and second highest value. 
#     # If there is a large difference (> 15%) between the max and second highest values, return just the max
#     # If MAX - Second_Highest < threshold then too close to tell -> return both values
#         
#     # List of return values. Either [Max index, Max value] or [Max index, Max value, Second Highest index, Second Highest value]
#     # If the image cannot be confidently classified, return a 0 to indicate, followed by value and prediction
#     return_list = []
#     
#     #Threshold value, can be tuned
#     threshold = 0.15    
#     
#     # Find the confidence of the max value, and it's index
#     max_value = max(confidence_values)
#     max_index = np.where(confidence_values == np.amax(confidence_values))[0][0]
#     
#     # If the best possible value (max) is less than 20%, the image cannot be classified confidently
#     if max_value > 0.20:
#             
#         # Create a copy of the values, and remove the previous max from the copy
#         temp_list = confidence_values.copy()
#         temp_list = np.delete(temp_list, np.where(temp_list == np.amax(confidence_values)))    
#         
#         # Find the confidence of the second highest value, and it's index in the ORIGINAL list of confidence values
#         second_highest_value = max(temp_list)
#         second_highest_index = np.where(confidence_values == second_highest_value)[0][0]
#         
#         
#         if (max_value - second_highest_value > threshold):
#             return_list.append([max_index, max_value])
#         
#         else:
#             return_list.append([max_index, max_value, second_highest_index, second_highest_value])
#             
#         return return_list
#     
#     else:
#         return_list.append([999, max_value, max_index])
#         return return_list
# 
# # Prediction test
#         
#     
# # =============================================================================
# # Gammel måte
# # test_dir = 'input/placeholder'
# # test_imgs = ['input/placeholder/{}'.format(i) for i in os.listdir(test_dir)]   # Get test images    
# #test_imgs.remove('centered/test/Thumbs.db')
# # =============================================================================
# 
# 
# # Ny måte
# tripple_db_path = '\\\\129.242.140.132\\remote\\UtklippsDatabaser\\full_3digit_trainingset.db'
# db = DbHandler(tripple_db_path)
# table = 'cells'
# data = db.select_all_training_images_3digit(table)
# test_imgs = []
# 
# for d in data:
#     image = decode_image(d[2])
#     
#     test_imgs.append(image)
# 
# random.shuffle(test_imgs)
#     
# #test_images = 200
# #X_test, y_test, image_names = read_and_process_image(test_imgs[:test_images], True) #y_test will in this case be empty
# X_test, y_test, image_names = read_and_process_image(test_imgs, True) #y_test will in this case be empty
# x = np.array(X_test)
# x = np.expand_dims(x, axis=-1)
# 
# test_datagen = ImageDataGenerator(rescale=1./255)
# 
# # Load model
# model = load_model('Centered_BalancedTraining_BalancedValidation_sparseCategoricalCrossentropy_ESValAcc_SimpleTrainGen_Model.HDF5')
#  
# # For loop to test our model
# i = 0
# plt.figure()
# 
# headers = ['Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5', 'Label 6', 'Label 7', 'Label 8', 'Label 9']
#   
# # We know that all the images are supposed to be 8 IN THIS CASE
# correct = 0
# almost_correct = 0
# 
# img_nr = 0
# for batch in test_datagen.flow(x, batch_size=1):
#     pred = model.predict(batch)                     # Pred is an array of 10 probability scores for each class (0-9). Softmax output
#     #pred = pred.reshape(pred.shape[1])
# 
#     text = ''
# 
#     for var in pred:
#         
#         confidence_values = evaluate_confidence_scores(var)
#         
#         for values in confidence_values:
#             
#             # Check if image was able to be classified confidently (With a confidence score above 20%)
#             if values[0] == 999:
#                 text = 'This image could not be classified with a confidence of over 20%, achieveing only ' + str( '%.4f' % (values[1] * 100)) + '% confidence as ' + str(values[2])
#                 break
#             
#             # Print out the classification and confidence score
#             table = tabulate(pred, headers, tablefmt='fancy_grid')
#             
#             text = 'Image number ' + str(img_nr) + ' was classified as ' + str(values[0]) + ' with a confidence score of ' + str( '%.4f' % (values[1] * 100)) + '%'
# # =============================================================================
# #             if values[0] == 8: correct += 1
# #             
# #             if len(values) > 2:
# #                 text += '\It was also classified as ' + str(values[2]) + ' with a confidence score of ' + str( '%.4f' % (values[3] * 100)) + '%'
# #                 if values[2] == 8: almost_correct += 1
# # =============================================================================
#         
#         
#     imgplot = plt.imshow(batch[0].squeeze())
#     
#     plt.title(text)   
#     plt.show()
#     print(table)
#     
#     img_nr += 1
#     
# # =============================================================================
# #     i += 1
# # 
# #     if i % 200 == 0:
# #         print('The model predicted that the image was an 8 : ' + str(correct) + ' number of times.')
# #         print('The number of times 8 was the models second choice, was: ' + str(almost_correct) + ' number of times.')
# #         break
# # =============================================================================
#     
#     
# =============================================================================
