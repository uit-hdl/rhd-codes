# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:02:36 2021

@author: bpe043
"""

from inference import main

import sqlite3
import tensorflow as tf

    
    
    
# Too many images to load them all, so we are doing it in batches
start = 0
end = 50000
increase = 50000
result = True
batch_index = 0


# Get cursor from the database
db = sqlite3.connect("<Path_to_your_database>")
cur = db.cursor()

# Exclusion set
training_db = sqlite3.connect("<Path_to_your_training_set_database>")
exclusion_names = training_db.cursor().execute("<Select_training_images_names_query>").fetchall()
exclusion_set = [x[0] for x in exclusion_names]
    
# Prediction model
prediction_model = tf.keras.models.load_model("<Path_to_saved_model>", compile = False)

# While we still have images to classify
while result == True:

    result = main(batch_index, start, end, cur, prediction_model, exclusion_set)
    
    start += increase
    end += increase
    batch_index += 1
