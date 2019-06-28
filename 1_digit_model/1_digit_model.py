# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:02:30 2019

@author: bpe043
"""

import cv2
import numpy as np

import matplotlib.pyplot as plt

import os
import random
import gc



from sklearn.model_selection import train_test_split

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


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
        
# Function to return list over full-path image names
def listdir_fullpath(folder):
    return [os.path.join(folder, image) for image in os.listdir(folder)]
    
train_imgs = listdir_fullpath('input/balanced_centered/')

train_imgs = train_imgs

# Smaller set, used for testing
#train_imgs = train_dash[:100] + train_ones[:100] + train_twos[:100] + train_threes[:100] + train_fours[:100] + train_fives[:100] + train_sixes[:100] + train_sevens[:100] + train_eights[:100] + train_nines[:100]

random.shuffle(train_imgs)

X, y = read_and_process_image(train_imgs, False)

del train_imgs
gc.collect()

X = np.array(X)
y = np.array(y)

print('Shape of training images are:', X.shape)
print('Shape of labels are:', y.shape)

#Reshape X for later use in Keras, current shape is (XXXX, 100, 100), we want it to be (XXXX, 100, 100, 1)
X = np.expand_dims(X, axis=-1)
    
print('Shape of training images after reshaping are:', X.shape)

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)

print('Shape of training images are:', X_train.shape)
print('Shape of validation images are:', X_val.shape)
print('Shape of training labels are:', y_train.shape)
print('Shape of validation labels are:', y_val.shape)

del X, y
gc.collect()

# Get the length of the train and validation data
ntrain = len(X_train)
nval = len(X_val)

es = EarlyStopping(monitor = 'val_loss', mode = 'auto', verbose=1, patience = 5)
cp = ModelCheckpoint('BalancedTraining_BalancedValidation_sparseCategoricalCrossentropy_SimpleTrainGen_Model.HDF5', monitor = 'val_loss', mode = 'auto', save_best_only=True, verbose=1)


# Batch size
batch_size = 32

# Creating the model
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape = (100, 100, 1)))
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


# Generators
train_datagen = ImageDataGenerator(rescale=1./255)

val_datagen = ImageDataGenerator(rescale=1./255)    # We do not augment validation data, we only perform rescale

# Creating generators
train_generator = train_datagen.flow(X_train, y_train, batch_size = batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size = batch_size)


# Training our model
history = model.fit_generator(train_generator,
                              steps_per_epoch=ntrain // batch_size,
                              epochs = 14,
                              validation_data = val_generator,
                              validation_steps = nval // batch_size,
                              callbacks = [es, cp]
                              )

model.save('Centered_BalancedTraining_BalancedValidation_sparseCategoricalCrossentropy_NoES_SimpleTrainGen_Model.HDF5')


# Plotting the training and validation curves
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()

# Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()
