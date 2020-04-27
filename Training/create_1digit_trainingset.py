# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:31:24 2019

@author: bpe043
"""

import pandas as pd
import sqlite3

import sys
sys.path.insert(0, '//homer.uit.no/bpe043/Desktop/Test_Projects/HelperFunctions')

from GND_Clustering import ImageSegClustering as splitter
from Database.dbHandler import DbHandler

import color_convert as cc
import sheer_image as sheer
import db_image_decode as decode

# OUTPUT database
db_loc = '\\\\129.242.140.132\\remote\\UtklippsDatabaser\\full_1digit_trainingset.db'
db_output = DbHandler(db_loc, validate=True)

def split_3digit_into_1digit_training(output_db):
    
    splitting_error = 0
    images_completed = 0
    
    conn = sqlite3.connect('\\\\129.242.140.132\\remote\\UtklippsDatabaser\\full_3digit_trainingset.db')
    
    query = 'SELECT * FROM cells'
    df = pd.read_sql_query(query, conn)
    
    df = df[['name', 'original', 'row', 'code', 'source']]
    
    total_images = len(df)

    
    # Iterate over each row in the dataframe, to get needed information from the original 3-digit images that will be split
    for index, row in df.iterrows():
        name = row['name']
        image= row['original']
        image_row = row['row']
        code= list(row['code'])     # To get easy access to each individual digit
        source = row['source']
        
        # Convert the image into a numpy array instead of a bytes-object
        image = decode.decode_image(image)
        
        # Get the split versions of the cell image, and all the different conversions
        split_result = splitter.split_and_convert(image)

        # If a split image exists in the 'split_orig' table, then it will also exist in the other cell tables
        if db_output.test_exists_any_source(name, 'split_orig'):
            images_completed += 1
            perc_done = ((images_completed + splitting_error) / total_images) * 100
            
            print('Skipping image {} that already exists in the database, - {}% done'.format(name, perc_done))
            continue        
        # Check if an error occured during the splitting of the image
        if split_result is None:
            splitting_error += 1
            with open('splitting errors.txt', 'a') as file:
                file.write('Error number: {} - Cell image: {} - Original image: {}\n\n'.format(str(splitting_error), name, source))
                
            perc_done = ((images_completed + splitting_error) / total_images) * 100
            print('An error occured with splitting the cell image: {} - From the original image: {}, - {}% done'.format(name, source, perc_done))
            continue
        
        
        i = 0
        while i < 3:
            split_name = code[i] + '-' + str(i) + '-' + name
            split_imgs = [split_result[x][i] for x in range(3)]
            
            
            # Else, upload the split images
            output_db.store_single_splits_training(split_name, split_imgs, image_row, str(i), code[i], len(code), name)
            i += 1
            
        perc_done = ((images_completed + splitting_error) / total_images) * 100
        print('Completed image {}, - {}% done'.format(name, perc_done))
        images_completed += 1
        
    
    conn.close()
    
    return df
    
    
df = split_3digit_into_1digit_training(db_output)