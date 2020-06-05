# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:33:14 2020

@author: bpe043
"""

import numpy as np
from keras.models import load_model
import pandas as pd
import cv2

import sys
sys.path.insert(0, '//homer.uit.no/bpe043/Desktop/Test_Projects/HelperFunctions')
from db_image_decode import decode_image
from Database.dbHandler import DbHandler

def create_exclusion_set():
    
    print('Creating exclusion set')
    
    # Exclude images from both the training set, and images from the full column that is empty/a line image
    with open('v3_remove_list.txt', 'r') as file:
        remove_list = file.readlines()
        
    # Due to the format of the remove_list, we need to rework it slightly
    remove_list = [x.split(',')[0] for x in remove_list]
    
    # We also turn it into a set for easier use later
    remove_list = set(remove_list)
        
    three_digit_trainingset = '\\\\129.242.140.132\\remote\\UtklippsDatabaser\\full_3digit_trainingset.db'
    db_trainingset = DbHandler(three_digit_trainingset)
    training_images = db_trainingset.select_all_image_names_any('cells')
    training_images = [train_img for t in training_images for train_img in t]               # Convert list of tuples to list
    training_images= set(training_images)
    
    exclusion_set = training_images.union(remove_list)
    
    print('Exclusion set has been created')
    
    return exclusion_set


def process_batch(db, exclusion_set, start, end):
    
    print('Getting batch from database')
    
    # Get all the images from the column
    names_and_images = db.select_name_and_images_any_batch('fields', start, end)
    
    # Check if we have processed all the images in the database
    if len(names_and_images) == 0:
        return None
    
    print('Removing uneligible images')
    
    names = [x[0] for x in names_and_images]
    
    # Remove all the indexes from names_and_images where the image is in the exclusion_set
    keep_indexes = []
    index = 0
    for n in names:
        if n not in exclusion_set:
            keep_indexes.append(index)
        
        index += 1
        
    print('Finished the process of excluding images from batch')
    
    keep_images = [names_and_images[i][1] for i in keep_indexes]
    keep_names = [names_and_images[i][0] for i in keep_indexes]
    
    keep_images = np.array(keep_images)
    keep_names = np.array(keep_names)
    
    return (keep_names, keep_images)

def decode_and_convert(names_and_images, batch_number):
    
    print('Starting conversion and logging of batch number {}'.format(batch_number))
    
    # Need to convert the images into the correct format
    # TODO: This functionality could probably be added to the utklipp function
    width = 200
    height = 115
    dim = (width, height)
    
    names = names_and_images[0]
    images = names_and_images[1]
    
    #For testing
    index = 0
    total = len(images)
    
    decoded_images = []
    
    # Decode the images
    for img in images:
        decoded = decode_image(img)
        decoded = cv2.resize(decoded, dim, interpolation = cv2.INTER_AREA)
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
        decoded_images.append(decoded)

        print('Complated number {} out of {}.'.format(index, total)) 
        index += 1
    
    #names_and_images.to_csv('C:\Production Images\\batch_{}.csv'.format(batch_number), sep = ';', index = False)
    decoded_images = np.array(decoded_images)
    np.save('C:\Production Images\\batch_{}_names'.format(batch_number), names)
    np.save('C:\Production Images\\batch_{}_images'.format(batch_number), decoded_images)


def three_digit_production(batch_number):
    
    print('Starting prediction of images')
    
    # images will be collected from the database prior, and "bad" images will have been removed
    # "Bad" here meaning empty, line images, and images from the training set
    batch_names = np.load('C:\Production Images\\batch_{}_names.npy'.format(batch_number))
    batch_images = np.load('C:\Production Images\\batch_{}_images.npy'.format(batch_number))
    batch_images = np.expand_dims(batch_images, axis=-1)

    # Still need the mapping
    mapping = np.load('C:\\Models\\Ground_truth_arrays\\3_digit_ground_truth_mapping.npy')

    model = load_model('C:\\Models\\Stratified_model_3-digit_greyscale_fold_9.h5')

    # Run prediction on the model with the full set of images
    
    # Confidence scores
    conf_scores = model.predict(batch_images)

    # Labels
    labels = np.argmax(conf_scores, axis = 1)
    labels = [mapping[x] for x in labels]

    # Getting confidence scores to csv
    conf = pd.DataFrame(data = conf_scores[0:, :-1],
                        index = range(0, len(conf_scores)),
                        columns = ['Class ' + c for c in [str(mapping[x]) for x in range(0, 264)] ])
    
    conf['Predicted label'] = labels
    conf['Image name'] = batch_names
    
    print('Completed prediction. Starting saving of results')
    
    # The labels and Confidence Scores for the entire column
    conf.to_csv('C:\\Production images\\Production_results\\batch_{}_results.csv'.format(batch_number), sep = ';', index=False)
    
    print('Results saved')
     
    return (labels, conf)


def main():

    # Can't work on the entire dataset column at once due to lack of RAM, so we are splitting it up into batches
    
    # Boolean flag to indicate if we have processed all the images in the database
    ended = False
    
    # Starting value for the batch
    start = 0
    
    # Ending value for the batch
    end = 500000
    
    # Increase value for start and end values
    increase = 500000
    
    # Database for the entire column
    full_dataset = '\\\\129.242.140.132\\remote\\UtklippsDatabaser\\3digit_Occupational_Codes_All.db'
    db = DbHandler(full_dataset)
    
    # Exclusion set
    exclusion_set = create_exclusion_set()
    
    batch_number = 0
    while ended is False:
        names_and_images = process_batch(db, exclusion_set, str(start), str(end))
        #names_and_images = True
        # Check to see if we have reached the end
        if names_and_images is None:
            ended = True
            break
    
        decode_and_convert(names_and_images, str(batch_number))
        three_digit_production(batch_number)
        batch_number += 1
        
        start += increase
        end += increase
    
    
main()








































