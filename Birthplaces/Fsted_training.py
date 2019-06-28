# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 14:46:23 2019

@author: bpe043
"""

from sklearn.model_selection import train_test_split

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

import numpy as np
import gc
import pickle

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

with open('birthplace_images.pkl', 'rb') as f:
    X = pickle.load(f)

with open('birthplace_labels.pkl', 'rb') as f:
    y = pickle.load(f)

X = np.array(X)
y = np.array(y)

print('Shape of training images are:', X.shape)
print('Shape of labels are:', y.shape)

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

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape = (210, 280, 3)))
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

model.add(layers.Dense(25, activation='softmax'))     


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