# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:50:57 2019

@author: bpe043
"""

import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


import sys
sys.path.insert(0, '//homer.uit.no/bpe043/Desktop/Test_Projects/HelperFunctions')

from Database.dbHandler import DbHandler
from db_image_decode import decode_image

def create_X_y(data, db, table, color):
    X = []
    y = []
    
    for digit in data:
        
        image = decode_image(digit[0])
        
        # If any original color images managed to sneak into the database
        if len(image.shape) > 2 and color != 'orig' and color != 'original':
            name = digit[1]
            db.remove_by_name(table, name)
            continue
        
        # Should have done the transformation into bitwise_not when uploading the images to the Black & White database
        if color == 'bw':
            image = cv2.bitwise_not(image)
            
        #TODO: Move this functionality. 3-digit images did not get standardized on upload, do that manually here for now. 
        if table == 'cells':
            width = 200
            height = 115
            dim = (width, height)
            
            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            
        X.append(image)
        
        label = digit[1]
        y.append(label)
        
        
    # Remove invalid labels and images
    zero_indexes = [i for i in y if i == '0']
    del y[: len(zero_indexes)]
    del X[: len(zero_indexes)]
        
    X = np.array(X)
    y = np.array(y)
    
    # If we are working with 3-digit codes, we need to remap each code to a number between 0-<max number of unique codes> to fit in our model's softmax output layer
    if table == 'cells':
        unique_labels = np.unique(y)
        temp = {y:x for x, y in enumerate(unique_labels)}
        
        y = [temp.get(elem) for elem in y]
        
        y = np.array(y)
        
        np.save('C:\\Models\\Ground_truth_arrays\\3_digit_{}_ground_truth_mapping'.format(color), unique_labels)

    
    # Reshape X for later use in Keras, normal shape is (XXXX, 100, 100) we want it to be (XXXX, 100, 100, 1) for B&W and Greyscale, for original images no expansion is needed
    if color != 'original':
        if color != 'orig':
            X = np.expand_dims(X, axis=-1)
        

        
    return X, y

def create_3digit_model(channels):
    # Creating the model
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape = (115, 200, channels)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    
    model.add(layers.Dropout(0.5))  # Dropout for regularization
    
    model.add(layers.Dense(512, activation='relu'))
    
    model.add(layers.Dense(256, activation='relu'))
    
    # When you have digits xxx - 264
    model.add(layers.Dense(265, activation='softmax'))     
    
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.adam(lr=0.001), metrics=['acc'])
    
    return model

def create_model(channels):
    # Creating the model
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape = (100, 100, channels)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    
    model.add(layers.Dropout(0.5))  # Dropout for regularization
    
    model.add(layers.Dense(512, activation='relu'))
    
    # When you have digits 1-9
    model.add(layers.Dense(10, activation='softmax'))     
    
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.adam(lr=0.0001), metrics=['acc'])
    
    return model

def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    
    return model.score(X_test, y_test)


# Currently working on 1 digit or 3 digit
three_digit = True


if three_digit is False:    
    # 1-digit
    #singles_db_path = '\\\\129.242.140.132\\remote\\UtklippsDatabaser\\full_1digit_trainingset.db'
    singles_db_path = '\\\\129.242.140.132\\remote\\UtklippsDatabaser\\full_1digit_trainingset_augmented.db'
    db = DbHandler(singles_db_path)
    table = 'split_{}'
    #colors = ['grey', 'bw', 'orig']
    colors = ['grey']
    digits = 1
    
else:
    # 3-digit
    tripple_db_path = '\\\\129.242.140.132\\remote\\UtklippsDatabaser\\full_3digit_trainingset.db'
    db = DbHandler(tripple_db_path)
    table = 'cells'
    digits = 3
    #colors = ['greyscale',  'black_white', 'original']
    #colors = ['greyscale']
    colors = ['original'] # For the greound truth that will be used by both 3 and 1 digit models
    validationMetric = 'val_acc'
    
    
    
for color in colors:
    
    if color == 'original' or color == 'orig':
        channels = 3
    else:
        channels = 1
        
    if table != 'cells':
        table = table.format(color)
        data = db.select_all_training_images(table)
        print('Got the data from the database')
        X, y = create_X_y(data, db, table)
        print('Transformed the data into training set and labels')
        
        # Use X and y to train the model, store X_val and y_val for validation later on in validation.py
        X, X_val, y, y_val = train_test_split(X, y, test_size = 0.20, stratify=y, shuffle = True)
        
        #np.save('C:\\Models\\Ground_truth_arrays\\1_digit_{}_ground_truth_images'.format(color), X_val)
        #np.save('C:\\Models\\Ground_truth_arrays\\1_digit_{}_ground_truth_labels'.format(color), y_val)
        np.save('C:\\Models\\Ground_truth_arrays\\1_digit_{}_ground_truth_images_all_digits_augmented'.format(color), X_val)
        np.save('C:\\Models\\Ground_truth_arrays\\1_digit_{}_ground_truth_labels_all_digits_augmented'.format(color), y_val)
        
    else:
        data = db.select_all_training_images_3digit(color, table)
        print('Got the data from the database')
        X, y = create_X_y(data, db, table, color)
        print('Transformed the data into training set and labels')
        
        # Use X and y to train the model, store X_val and y_val for validation later on in validation.py
        X, X_val, y, y_val = train_test_split(X, y, test_size = 0.20, shuffle = True)
        
        np.save('C:\\Models\\Ground_truth_arrays\\3_digit_{}_ground_truth_images'.format(color), X_val)
        np.save('C:\\Models\\Ground_truth_arrays\\3_digit_{}_ground_truth_labels'.format(color), y_val)

    

    # Batch size
    batch_size = 32
    
    """ K-fold """
    
    splits = 10
    fold = 0
    
    # Clear model and create it
    model = None
    if digits == 1:    
        model = create_model(channels)
    else:
        model = create_3digit_model(channels)
    
    #Instantate the cross validator
    skf = StratifiedKFold(n_splits = splits, shuffle = True)
    
    for index, (train_indices, val_indices) in enumerate(skf.split(X, y)):
        print('Training on fold {}/10 for {}...'.format(index+1, color))
        
        xtrain, xval = X[train_indices], X[val_indices]
        ytrain, yval = y[train_indices], y[val_indices]
        
        print('Training new interation on {} training samples, {} validation samples. This may be a while...'.format(str(xtrain.shape[0]), str(xval.shape[0])))
        
        history = model.fit(xtrain, ytrain, batch_size = batch_size, epochs = 10)
        
        evaluate = model.evaluate(xval, yval)
        print('Model evaluation ', evaluate)
        
        accuracy_history = history.history['acc']
        #val_accuracy_history = history.history['val_accuracy']
        
        print('Last training accuracy: {}'.format(str(accuracy_history[-1])))
        
        with open('Models2\\Results.txt', 'a') as file:
        #with open('Models2\\1-digit\\Results.txt', 'a') as file:
# =============================================================================
#             file.write('Training accuracy for fold {} for {} was : {}\n'.format(index, color, accuracy_history[-1]))
#             file.write('Validation accuracy for fold {} for {} was : {}\n\n'.format(index, color, evaluate[-1]))
# =============================================================================
            file.write('Training accuracy for augmented fold {} for {} was : {}\n'.format(index, color, accuracy_history[-1]))
            file.write('Validation accuracy for augmented fold {} for {} was : {}\n\n'.format(index, color, evaluate[-1]))
            
        #model.save('C:\\Models\\Stratified_model_{}-digit_{}_fold_{}.h5'.format(digits, color, index))
        model.save('C:\\Models\\Stratified_augmented_model_{}-digit_{}_fold_{}.h5'.format(digits, color, index))
    

