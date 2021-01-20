
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:35:33 2020

@author: bpe043
"""


import cv2


import sys
sys.path.insert(0, '<path to helper functions here>')
from db_image_decode import decode_image
from Database.dbHandler import DbHandler
from GND_Clustering import ImageSegClustering as splitter

from color_convert import convert_img_bw

###############################################################################
###############################################################################
""" Path to the folder containing the helper functions need to be added at line 14 """
""" Path to database with training set images needs to be added at line 290 """
""" And the table name from the database should be added to the list at line 286"""
###############################################################################
###############################################################################


def create_exclusion_set(db_trainingset):
    
    print('Creating exclusion set')
    
    # Exclude images from both the training set, and images from the full column that is empty/a line image
    with open('v4_remove_list.txt', 'r') as file:
        remove_list = file.readlines()
        
    # Due to the format of the remove_list, we need to rework it slightly
    remove_list = [x.split(',')[0] for x in remove_list]
    
    # We also turn it into a set for easier use later
    remove_list = set(remove_list)

    training_images_names = db_trainingset.select_all_image_names_any('cells')
    training_images_names = [train_img for t in training_images_names for train_img in t]               # Convert list of tuples to list
    training_images_names= set(training_images_names)
    
    exclusion_set = training_images_names.union(remove_list)
    
    print('Exclusion set has been created')
    
    return exclusion_set



def process_batch(db, start, end, table, training, exclusion_set = None, special = False, image_column = None, oneDigit = False, augmenting = False):
    
    print('Getting batch from database')
    
    # Get all the images from the column, as well as their name and possibly labels
    if training is True and special == False:
        
        if oneDigit == True:
            names_and_images = db.select_name_images_and_labels_any_batch_1digit(table, start, end)
        elif augmenting == True:
            names_and_images = db.select_name_images_and_labels_any_batch_augmented(table, start, end)
        else:
            names_and_images = db.select_name_images_and_labels_any_batch(table, start, end)
            
    elif training is False and special == False:
        names_and_images = db.select_name_and_images_batch(table, start, end)
    else:
        names_and_images = db.select_name_and_images_any_batch(table, image_column, start, end)

    
    # Check if we have processed all the images in the database
    if len(names_and_images) == 0:
        return None
        
    # Differentiate between training and production
    
    # Production
    if exclusion_set is not None:
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
        
        return (keep_names, keep_images)
        
    # Training
    elif exclusion_set is None and special == False:
        
        keep_images = []
        keep_names = []
        keep_labels= []
        
        if augmenting == True:
            keep_sources = []
        
        for i in range(len(names_and_images)):
            
            keep_images.append(names_and_images[i][1])
            keep_names.append(names_and_images[i][0])
            keep_labels.append(names_and_images[i][2])
            
            if augmenting == True:
                keep_sources.append(names_and_images[i][3])
                
                
        if augmenting == True:
            return (keep_names, keep_images, keep_labels, keep_sources)
        else:
            return (keep_names, keep_images, keep_labels)
    
    # Special
    else:
        keep_images = []
        keep_names = []
        for i in range(len(names_and_images)):
            
            keep_images.append(names_and_images[i][1])
            keep_names.append(names_and_images[i][0])
            
        # Remove 65 extra images from the training set to make sure we have a completely even split
        if len(keep_images) > 35000:
            keep_images = keep_images[0:35000]
            keep_names= keep_names[0:35000]
            
        return (keep_names, keep_images)
                
    

def decode(names_and_images, batch_number, oneDigit = False, training = False):
    
    print('Starting conversion of batch number {}'.format(batch_number))
    
    # Need to convert the images into the correct format
    # TODO: This functionality could probably be added to the utklipp function
    
    # If we want to use 1 digit images, use this. If not, there is no reason to differentiate the sizes
# =============================================================================
#     if training == True:
#         width = 200
#         height = 115
#     else:
#         width = 100
#         height = 100
# =============================================================================
    
    width = 200
    height = 115
        
    dim = (width, height)
    
    names = names_and_images[0]
    images = names_and_images[1]
    
    index = 0
    total = len(images)
    
    decoded_images = []
    decoded_names = []
    
    splittable_images = []
    
    non_splittable_images = []
    non_splittable_names = []
    
    
    # Decode the images
    for img in images:
        decoded = decode_image(img)
        
        if oneDigit == True and training == False:
            decoded_split = splitter.split_and_convert(decoded, onlyGrey = True)
            
            if decoded_split is not None:
                split_image = (decoded_split[0], decoded_split[1], decoded_split[2])
                splittable_images.append(split_image)
                
            else:
                non_splittable_images.append(decoded)
                non_splittable_names.append(names[index])
                
        if training == False or len(decoded.shape) > 2:
            decoded = cv2.resize(decoded, dim, interpolation = cv2.INTER_AREA)
            #decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY) 
            decoded = convert_img_bw(decoded)
            
        decoded_images.append(decoded)
        decoded_names.append(names[index])


        print('Complated decoding and conversion of image number {} out of {}.'.format(index, total)) 
        index += 1


        
    if oneDigit == False or training == True:
        return (decoded_names, decoded_images)
    else:
        return (decoded_names, decoded_images, splittable_images, non_splittable_names, non_splittable_images)
    


def get_images(db_fullset, db_trainingset, table_list, training, start, end, batch_number, special = False, oneDigit = False, augmenting = False):
    
    labels = []
    sources = []
    decoded_images = []
    decoded_names = []
    
        
    # For training
    if training == True and special == False:
        db = db_trainingset
        
        if oneDigit == True:
            names_and_images = process_batch(db, str(start), str(end), table_list[2], True, oneDigit = oneDigit)
        else:
            names_and_images = process_batch(db, str(start), str(end), table_list[0], True, augmenting = augmenting)
        
        decoded_names, decoded_images = decode(names_and_images, batch_number, training = training)
            
        labels += names_and_images[2]
        
        if (augmenting == True):
            sources += names_and_images[3]
            return ((decoded_names, decoded_images), labels, sources)
        else:
            return ((decoded_names, decoded_images), labels)
    
    # For produciton
    elif training == False and special == False:
        db = db_fullset
        # Exclusion set
        exclusion_set = create_exclusion_set(db_trainingset)
        names_and_images = process_batch(db, str(start), str(end), table_list[1], False, exclusion_set)
        
        # Once we get back None, we've exhausted all images, and can now begin Production
        if names_and_images is None:
            return None
        else:
            #decoded_names, decoded_images = decode(names_and_images, batch_number, oneDigit)            
            #return(decoded_images, decoded_names)
            return(decode(names_and_images, batch_number, oneDigit))
        
    # For the special case
    else:
        db_1 = db_trainingset
        db_2 = db_fullset
        
        names_and_images_1 = process_batch(db_1, str(start), str(end), table_list[0], training, special = True, image_column = 'Original')
        names_and_images_2 = process_batch(db_2, str(start), str(end), table_list[1], training, special = True, image_column = 'Image')
    
        combined_images = names_and_images_1[1] + names_and_images_2[1]
        combined_names = names_and_images_1[0] + names_and_images_2[0]
        
        names_and_images = (combined_names, combined_images)
        
        decoded_names, decoded_images = decode(names_and_images, batch_number)
        
        label_list_1 = [1] * len(names_and_images_1[0])
        label_list_2 = [0] * len(names_and_images_2[0])
        labels += label_list_1 + label_list_2
        
        return((decoded_names, decoded_images), labels)
        


def run(training, start, end, batch_number, special = False, oneDigit = False, augmenting = False):
    
    """ insert name of the table containing the training images in this list """
    table_list = ['<Table name where the training images are stored>']
    
    # Training set database
    if oneDigit == False:
        trainingset = '<path to database containing training images here!>'
        
    else:
        trainingset = 'full_1digit_trainingset.db'
        
    db_trainingset = DbHandler(trainingset)
    
    # Database for the entire column
    if special == False:
        full_dataset = '3digit_Occupational_Codes_All_longer.db'    # production
        #full_dataset = 'mini_subset.db'         # Compare prod and test images
    else:
        full_dataset = 'Production_Occupational_Subset.db'         # Compare prod and test images
    db_fullset = DbHandler(full_dataset)
    
    

    result = get_images(db_fullset, db_trainingset, table_list, training, start, end, batch_number, special, oneDigit, augmenting)
        

    return result
