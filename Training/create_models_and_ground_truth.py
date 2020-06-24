# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:50:57 2019

@author: bpe043
"""

import cv2
import numpy as np
import pandas as pd

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
        
        
# =============================================================================
#     # Remove invalid labels and images
#     zero_indexes = [i for i in y if i == '0']
#     del y[: len(zero_indexes)]
#     del X[: len(zero_indexes)]
# =============================================================================
        
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

def data_augmentation_create_X_y(data):
    
    digit = 1
    decoded_images = []
    while digit < 10:
        #images = db.select_by_actual_digit('split_grey', str(digit))    
        images = [d for d in data if d[6] == str(digit)]
        
        for image in images:
            
            # Decode the images
            im = decode_image(image[2])
            im = np.expand_dims(im, axis=-1)    # Used for greyscale and black and white, if you want to do this for color images, this line is not needed
            
            # Get the original variables that are needed to upload the augmented images back into the training set
            orig_name = image[1]
            row = image[4]
            pos = image[5]
            actual_digits = image[6]
            number_of_digits = image[7]
            source = image[8]
            
            # Make sure only valid, greyscale images, are kept. If working with color images, this check is not needed
            if len(im.shape) < 4:
                decoded_images.append( (im, orig_name, row, pos, actual_digits, number_of_digits, source) )
                
        
        digit += 1
        
    return decoded_images
                
                

def data_augmentation_augment_and_upload(decoded_images, iterations):
    
    from imgaug import augmenters as iaa
    import sqlite3
    
        
    out_path = 'C:\\DB\\augmented_only_training_1digit.db'
    conn = sqlite3.connect(out_path)
    c = conn.cursor()
    
    # Container for ALL the augmented images, that will go on to be used in training our model
    all_augmented_images = []
    all_augmented_images_labels = []
    
    # Define how many images you want to create
    # Each iteration is a doubling of the original set of images. 
    # If you have 1000 images, setting iterations = 4 will net you 4000 images
    j = 0
    end = iterations
    while j < end:

        # The transformations / augmentations we want to perform on the images
        seq = iaa.Sequential([
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            ),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            iaa.Crop(percent=(0, 0.1)),
            iaa.LinearContrast((0.75, 1.5)),
            iaa.Multiply((0.6, 1.1), per_channel=0.2),
                ], random_order = True)
    
        
        augmented_images = seq(images= [x[0] for x in decoded_images])
        
        insert_query = """INSERT INTO images (original, augmented, orig_name, row, position, actual_digits, number_of_digits, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?) """
        
        # Convert both images to binary
        i = 0
        while i < len(augmented_images):
            original = cv2.imencode('.jpg', decoded_images[i][0])[1]
            augmented = cv2.imencode('.jpg', augmented_images[i])[1]
            orig_name = 'augmented_' + str(j) + '_' + decoded_images[i][1]
            row = decoded_images[i][2]
            position = decoded_images[i][3]
            actual_digits = decoded_images[i][4]
            number_of_digits = decoded_images[i][5]
            source = decoded_images[i][6]
            
            
            insert_tuple = (original, augmented, orig_name, row, position, actual_digits, number_of_digits, source)
            c.execute(insert_query, insert_tuple)
            conn.commit()
            print('Completed image number {} out of {} in iteration {} out of {}'.format(str(i), len(augmented_images), j+1, end))
            
            # In addition to storing the augmented training set we made in a database, we also store the image, alone, in our container, that will be returned and used in training the model
            all_augmented_images.append(augmented_images[i])
            all_augmented_images_labels.append(actual_digits)
            
            i += 1
        
        j += 1 
            
        
    c.close()
    return all_augmented_images, all_augmented_images_labels



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
    
    model.add(layers.Dense(10, activation='softmax'))     
    
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.adam(lr=0.0001), metrics=['acc'])
    
    return model

def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    
    return model.score(X_test, y_test)


# Currently working on 1 digit or 3 digit
three_digit = False


if three_digit is False:    
    # 1-digit
    singles_db_path = '\\\\129.242.140.132\\remote\\UtklippsDatabaser\\full_1digit_trainingset.db'
    db = DbHandler(singles_db_path)
    table = 'split_{}'
    #colors = ['grey', 'bw', 'orig']
    colors = ['grey']
    digits = 1
    
    # Augment the training set, but not the validation set
    augment = False
    iterations = 4
    
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
        
        if augment is True:
            data = db.select_all_images_any(table)
            print('Got the data from the database')
            
            data_list = data_augmentation_create_X_y(data)
            
            # Once we have the data from the images, we need to make our X and y
            X_images = np.array([i[0] for i in data_list])
            y = np.array([i[4] for i in data_list])
        
            # To keep track of which images are part of the augmented training set, and which will be part of the non-augmented validation set, we need to do some trickery
            fake_X_indeices = list(range(0, len(y)))
            
            # Here, X_indices will be the indices of the images in x that we will pass on to our augment function
            # X_val_indices will be the indices of the images in x that will be part of our non-augmented validation set
            X_indices, X_val_indices, y, y_val = train_test_split(fake_X_indeices, y, test_size = 0.20, stratify=y, shuffle = True)
        
            # "Select" images based on the indices given to us by the train test split
            X = [data_list[x] for x in X_indices]
            X_val = np.array([X_images[x] for x in X_val_indices])
            
            print('Transformed the data into training set and labels')
            
            # Then we store our validation set for use later on in testing
            np.save('C:\\Models\\Ground_truth_arrays\\1_digit_{}_ground_truth_images_only_augmented_training'.format(color), X_val)
            np.save('C:\\Models\\Ground_truth_arrays\\1_digit_{}_ground_truth_labels_only_augmented_training'.format(color), y_val)
            
            # Then we do our augmentation of the trainingset
            X, y = data_augmentation_augment_and_upload(X, iterations)
            
            X = np.array(X)
            y = np.array(y)
            
            print('Augmented the training set')
            
            
        else:
            data = db.select_all_training_images(table)
            print('Got the data from the database')
            X, y = create_X_y(data, db, table, color)
            print('Transformed the data into training set and labels')
        
        
            # Use X and y to train the model, store X_val and y_val for validation later on in validation.py
            X, X_val, y, y_val = train_test_split(X, y, test_size = 0.20, stratify=y, shuffle = True)

            np.save('C:\\Models\\Ground_truth_arrays\\1_digit_{}_ground_truth_images_all_digits'.format(color), X_val)
            np.save('C:\\Models\\Ground_truth_arrays\\1_digit_{}_ground_truth_labels_all_digits'.format(color), y_val)
            
        
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
            file.write('Training accuracy for augmented only-training fold {} for {} was : {}\n'.format(index, color, accuracy_history[-1]))
            file.write('Validation accuracy for augmented only-training fold {} for {} was : {}\n\n'.format(index, color, evaluate[-1]))
            
        #model.save('C:\\Models\\Stratified_model_{}-digit_{}_fold_{}.h5'.format(digits, color, index))
        model.save('C:\\Models\\Stratified_all_digits_model_{}-digit_{}_fold_{}.h5'.format(digits, color, index))
    
#        predictions = model.predict(xval)

    
    # Previous code
# =============================================================================
#     kfold = StratifiedKFold(n_splits = splits, shuffle = True, random_state = 42)
#     
#     for train_index, test_index in kfold.split(X, y):
#     
#         #es = EarlyStopping(monitor = validationMetric, mode = 'auto', verbose=1, patience = 4)
#         #cp = ModelCheckpoint('Models\\K_fold\\Fold_{}_{}_Distribution_3digit_{}.HDF5'.format(fold, color, validationMetric), monitor = validationMetric, mode = 'auto', save_best_only=True, verbose=1)
#     
#         x_train, x_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         
#         model = create_model(channels)
#         history = model.fit(x_train, y_train, batch_size = batch_size, validation_split = 0.2, epochs = 25, callbacks = [es, cp])
#         
#         print('Model evaluation ', model.evaluate(x_test, y_test))
# 
#         
#         acc = history.history['acc'][-1]
#         loss = history.history['loss'][-1]
#         val_acc = history.history['val_acc'][-1]
#         val_loss = history.history['acc'][-1]
# 
#         with open('Results.txt', 'a') as file:
#             file.write('\nThe results of Fold {} type {} was Acc: {} Loss: {} Val Acc: {} Val Loss: {}'.format(fold, color, acc, loss, val_acc, val_loss) )
#         
#         
#         
#         fold += 1
# =============================================================================
        
        
    
#, callbacks = [es, cp]
# =============================================================================
# 
# Batch size
#batch_size = 32
#
# Generators
#train_datagen = ImageDataGenerator(rescale=1./255)
#
#val_datagen = ImageDataGenerator(rescale=1./255) 
#    
# """ Train-test split """
# # Splitting into training and validation sets. Correlation between image(X) and digit/label (y) is maintained
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.20, stratify=y, random_state=2)
# 
# # Length of the training and validation data
# ntrain = len(X_train)
# nval = len(X_val)
# 
# # Creating generators
# train_generator = train_datagen.flow(X_train, y_train, batch_size = batch_size)
# val_generator = val_datagen.flow(X_val, y_val, batch_size = batch_size)
# 
# # Training our model
# history = model.fit_generator(train_generator,
#                               steps_per_epoch=ntrain // batch_size,
#                               epochs = 100,
#                               validation_data = val_generator,
#                               validation_steps = nval // batch_size,
#                               callbacks = [es, cp]
#                               )
# 
# 
# # Plotting the training and validation curves
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# 
# epochs = range(1, len(acc) + 1)
# 
# # Train and validation accuracy
# plt.plot(epochs, acc, 'b', label='Training accuracy')
# plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
# plt.title('Training and Validation accuracy')
# plt.legend()
# 
# plt.figure()
# 
# # Train and validation loss
# plt.plot(epochs, loss, 'b', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and Validation loss')
# plt.legend()
#     
# plt.show()
# =============================================================================
