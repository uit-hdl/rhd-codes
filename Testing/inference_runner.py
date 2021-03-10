# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:02:36 2021

@author: bpe043
"""

from inference import main

import sqlite3
import tensorflow as tf




""" Seems even doing inference in relatively small batches takes up too much memory currently.
    Therefore I'm trying to create an external script to handle the "Divide into batches and send each batch to inference" part, to allow the OS to reclaim the memory from the process """
    
    
    
    
    
# Too many images to load them all, so we are doing it in batches
start = 0
end = 50000
increase = 50000
result = True
batch_index = 0


# Get cursor from the database
db = sqlite3.connect('\\\\129.242.140.132\\remote\\UtklippsDatabaser\\3digit_Occupational_Codes_All_longer.db')
cur = db.cursor()

# Exclusion set
dugnad = sqlite3.connect('\\\\129.242.140.132\\remote\\UtklippsDatabaser\\dugnads_sett_no_u.db')
exclusion_names = dugnad.cursor().execute("SELECT Name FROM CELLS").fetchall()
exclusion_set = [x[0] for x in exclusion_names]
    
# Prediction model
prediction_model = tf.keras.models.load_model('dugnad_ctc_prediction', compile = False)

# While our start does not exceed the max number of images
while result == True:

    result = main(batch_index, start, end, cur, prediction_model, exclusion_set)
    
    start += increase
    end += increase
    batch_index += 1