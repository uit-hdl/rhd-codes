# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:35:28 2021

@author: bpe043
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 09:25:29 2020

@author: bpe043
"""

import numpy as np
import pandas as pd
import math
import random

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.models import load_model


from sklearn.metrics import classification_report

from keras import layers
from keras import models
from keras import optimizers
from keras.optimizers import SGD


""" Function where images are fetched from the database, reshaped and convert to the appropriate form and color"""
from get_images import run
    
    

def create_model(channels, n_classes, height = 115, width = 200):
    # Creating the model
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape = (height, width, channels)))
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
    #model.add(layers.Dense(512, activation='sigmoid'))
    
    model.add(layers.Dense(256, activation='relu'))
    #model.add(layers.Dense(256, activation='sigmoid'))
    
    # When you have digits xxx - 264
    model.add(layers.Dense(n_classes, activation='softmax'))  
    
    opt = 'adam'
    #opt = SGD(lr=0.0001)
    
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer= opt, metrics=['acc'])
    
    return model


def validate_training_model(validation_x, validation_y, image_names, model_name, version):
    
    mapping = np.load('{}_3_digit_ground_truth_mapping.npy'.format(version))
    
    actual_labels = [c for c in [str(mapping[x]) for x in validation_y] ]
    
    
    model = load_model(model_name)
    
    # Run prediction on the model with the ground truth images, to get the predicted labels
    y_pred = model.predict(validation_x)

    y_pred2 = np.argmax(y_pred, axis = 1)
    
    # Get Predicted labels
    y_pred_mapped = [mapping[x] for x in y_pred2]

    # Get the report
    report = classification_report(validation_y, y_pred2, output_dict = True)
    
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
    

    df['class'] = new_index
    columns =['class', 'precision', 'recall', 'f1-score', 'support']
    df = df[columns]

    
    print(df)

    # Getting confidence scores to csv
    conf = pd.DataFrame(data = y_pred[0:, :-1],
                        index = range(0, len(y_pred)),
                        columns = ['Class ' + c for c in [str(mapping[x]) for x in range(0, len(mapping))] ])
    conf['True_Label'] = actual_labels
    conf['Predicted_Label'] = y_pred_mapped
    
    if image_names != None:
        conf['Image_names'] = image_names
    
    conf.to_csv('C:\\Training_results\\{}_Training_evaluation_confidence_scores.csv'.format(version), sep = ';', index=False)

    df.to_csv('C:\\Training_results\\{}_Training_evaluation_classification_report.csv'.format(version), sep = ';', index=False)
    

    
    

    
validationMetric = 'val_acc'

# Get our data
training = True
version = 'dugnad'
channels = 1
random_shuffle = False
only_final = False

start = 0
end = 35000         # Random number larger than the total amount of images in the trainingset
batch_number = 0
data = run(training, start, end, batch_number)

X = data[0][1]
image_names = data[0][0]
y = data[1]


# Reshape the data to the appropriate form for our model
X = np.array(X)
y = np.array(y)

X = np.expand_dims(X, axis=-1)

unique_labels = np.unique(y)

# Number of classes
#n_classes = 265  # Usually
n_classes = len(unique_labels) + 1     # Dugnad


if training == True:
    np.save('{}_3_digit_ground_truth_mapping'.format(version), unique_labels)

temp = {y:x for x, y in enumerate(unique_labels)}

y = [temp.get(elem) for elem in y]

# For testing for data leaks by randomly shuffling the labels
if random_shuffle == True:
    random.shuffle(y)
    
new_y = []
i = 0
while i < len(y):
    new_y.append(str(y[i]) + ',' +  image_names[i])
    i += 1

y = np.array(new_y)

# The full dataset, correctly shaped. Will be used for training model_2 when we are creating a final training model that will not be used to evaluate
X_full = X
y_full = y


# Use X and y to train the model, use X_val and y_val to evaluate the model when training and evaluating a training model
X, X_val, y, y_val = train_test_split(X, y, test_size = 0.20, shuffle = True)

#model = load_model('Models\\dugnad_training.h5')
#pred = model.predict(X_val)


image_names_val = []
i = 0
while i < len(y_val):
    to_split = y_val[i]
    y_part = to_split.split(',')[0]
    name_part = to_split.split(',')[1]
    
    y_val[i] = int(y_part)
    image_names_val.append(name_part)
    
    i += 1
    
# Convert y back to the correct type
y_val = np.array(y_val).astype('int32')
    
image_names = []
i = 0
while i < len(y):
    to_split = y[i]
    y_part = to_split.split(',')[0]
    name_part = to_split.split(',')[1]
    
    y[i] = int(y_part)
    image_names.append(name_part)
    
    i += 1
    
# Convert y back to the correct type
y = np.array(y).astype('int32')



# Kfold_training
k = 10
color = 'b&w'

# Batch size
batch_size = 32



if only_final == True:
    n_epochs = 10
    model_2 = create_model(channels, n_classes)
    #model_2 = create_binary_model(channels)
    model_2.fit(X, y, batch_size = batch_size, epochs = n_epochs, validation_split = 0)
    
    final_model_name = 'Models\\{}_KFold_3digit_{}_epochs.h5'.format(version, color, n_epochs)

    model_2.save(final_model_name)

    # Only do this for when you want to validate your training model
    validate_training_model(X_val, y_val, image_names_val, final_model_name, version)
    #validate_binary_model(X_val, y_val, image_names, final_model_name)

else:
    skf = StratifiedKFold(n_splits = k, shuffle = True) # Shuffle provides randomized indices
    
    # These frames should have a length of K
    stopping = pd.DataFrame(None, columns = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9'])
    train_loss = pd.DataFrame(None, columns = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9'])
    
    
    for index, (train_indices, val_indices) in enumerate(skf.split(X, y)):  # For fold in folds
        
        # Clear model and create it
        model = create_model(channels, n_classes)
        
        print('Training on fold {}/10...'.format(index+1))
        
        xtrain, xval = X[train_indices], X[val_indices]
        ytrain, yval = y[train_indices], y[val_indices]
    
        print('Training new interation on {} training samples, {} validation samples. This may be a while...'.format(str(xtrain.shape[0]), str(xval.shape[0])))
        
        history = model.fit(xtrain, ytrain, batch_size = batch_size, epochs = 15, validation_data = (xval, yval) )
        
        stopping[stopping.columns[index]] = history.history['val_loss']
        train_loss[train_loss.columns[index]] = history.history['loss']
    
    # Plotting all k of the value loss per epoch 
    ax = stopping.plot(kind = 'line', legend = False, color = ['lightgrey'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation loss')
    
    
    # The expected (average) loss for each number of epochs
    mus = stopping.mean(numeric_only = True, axis = 1)
    
    # The standard deviation for each number of epochs
    sds = stopping.std(axis = 1, skipna = True)     # Gets standard deviation per columns
    ses = sds / math.sqrt(k-1)                      # Standard errors
    
    # Expected loss
    ax.plot(mus, color = 'black', linewidth = 2)
    
    # Confidence limits
    ax.plot(mus + 2*ses, color = 'black', linewidth = 1.2)
    ax.plot(mus - 2*ses, color = 'black', linewidth = 1.2)
    
    # Smallest loss
    mn = mus.idxmin()
    
    # Hastie and Tibshirani recommend taking the most conservative value (smallest number of epochs for us)
    # that is within one standard error of the minimum value
    upper = mus[mn] + ses[mn]
    lower = mus[mn] - ses[mn]
    se_1 = mus.between(lower, upper)
    se_2 = mus[se_1.values].idxmin() + 1 # Due to python starting it's count at 0
    
    # Shows minimum
    ax.axvline(x = mn, linestyle = 'dashed', linewidth = 2)
    
    # Shows the se_2 value (might be the same)
    # se_2 -1 to get back to the actual value, instead of the index value
    ax.axvline(x = se_2 - 1, linestyle = 'dashed', color = 'red', linewidth = 2)
    
    # Save fig
    #ax.figure.savefig('ValidationLoss_Epoch.png', dpi = 300)
    
    n_epochs = se_2
    
    model_2 = create_model(channels, n_classes)
    
    print('Performing final training with {} epochs on {} training samples. Almost done... hopefully'.format(str(n_epochs), str(X.shape[0])))
    
    
    # For temporary training, we run the training one final time on the final mode, with the correct number of epochs. But only on the 80% test data
    if training == True:
        model_2.fit(X, y, batch_size = batch_size, epochs = n_epochs, validation_split = 0)
        
        if random_shuffle == True:
            final_model_name = 'Models\\Random_KFold_3digit_{}_epochs.h5'.format(color, n_epochs)
        else:
            final_model_name = 'Models\\v{}_Training_KFold_3digit_{}_epochs.h5'.format(version, color, n_epochs)
    
        model_2.save(final_model_name)
    
        # Only do this for when you want to validate your training model
        validate_training_model(X_val, y_val, None, final_model_name, version)
    
    # For production, this should be done. When X and y are trainingset only
    # =============================================================================
    # elif training == True:
    #     history_2 = model_2.fit(X_full, y_full, batch_size = batch_size, epochs = n_epochs, validation_split = 0)
    #     final_model_name = 'Models\Production_KFold_stratified_model_3-digit_{}_{}_epochs.h5'.format(color, n_epochs)
    #     
    #     model_2.save(final_model_name)
    # 
    # =============================================================================