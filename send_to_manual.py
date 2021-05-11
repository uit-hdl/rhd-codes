# -*- coding: utf-8 -*-
"""
Created on Thu May  6 09:27:37 2021

@author: bpe043
"""

import pandas as pd
import numpy as np
import sqlite3

""" 
    Script to generate a list of the names of images that should be sent to manual validation and correction.
    Criteria for this is:
        1. Images with at least 1 out of 3 confidence scores below the chosen threshold of 65%
        2. Non-sensical codes (Predicted labels that are some combination of Text/Blank/Digit)
        3. Invalid codes (Not an actual valid 3-digit code found in the official list of codes used by Statistics Norway for the 1950 census)
        4. Possibly invalid codes (Codes that exists in the official list, but does not appear in the training set)
        Note that some overlap between these categories occur.
            
    With 'Text' images: Total amount of images sent to manual validation and correction should be 102.984 images (1.4% of the total amount of images)
    Without 'Text' images: Total amount of images sent to manual validation and correction should be 167.233 images (2.3% of the total amount of images)
"""

# Load in the predictions from our model
v = pd.read_csv('C:\\New_production_results\\CTC_dugnad\\total_confidence_scores.csv', sep = ';') # Full results
v = v[v.columns[1:]]
v_total = len(v)

# Images that were predicted to be 'bbb' or 'Blank' is automatically excluded from the rest of the results, as they contain no code to check
v = v[v.Predicted_Label != 'bbb']

# Load in the official list of codes from Statistics Norway
official_list = pd.read_csv('C:\\New_production_results\\CTC_dugnad\\1950_Occupational_Codes_list.csv', error_bad_lines = False, encoding = "ISO-8859-1", engine='python', sep = ';')
official_list = official_list.Code.drop_duplicates()

# Load in the codes from the training set
training = sqlite3.connect('\\\\129.242.140.132\\remote\\UtklippsDatabaser\\dugnads_sett_no_u.db')
training = training.cursor().execute("SELECT CODE FROM CELLS").fetchall()
training = [x[0] for x in training]
training = pd.Series(np.unique(training), name = 'Code')

# Output frame
to_manual = pd.DataFrame(columns = v.columns)

# Step 1
temp_frame = v[v.columns[-3:]]
temp_frame = temp_frame.loc[(temp_frame <= 0.65).any(1)]
temp_frame = v.loc[temp_frame.index]
to_manual = to_manual.append(temp_frame, ignore_index = True)
v = v.drop(temp_frame.index)

# Step 2
predictions = v.Predicted_Label
indexes = []

for index, s in predictions.iteritems():
    
    has_number = False
    has_t = False
    has_b = False
    
    has_number = any(char.isdigit() for char in s)
    has_t = 't' in s
    has_b = 'b' in s
    
    # Method of checking if more than one boolean var is true
    if sum(map(bool, [has_number, has_t, has_b])) > 1:
        indexes.append(index)

to_manual = to_manual.append(v.loc[indexes], ignore_index = True)
v = v.drop(indexes)

# Step 3
temp_frame = v[~v.Predicted_Label.isin(official_list)]

# For now, we keep images with the label 'Text', but this can be changed by un-commenting out the line below
#temp_frame = temp_frame.drop(temp_frame[temp_frame.Predicted_Label == 'ttt'].index)

to_manual = to_manual.append(temp_frame, ignore_index = True)
v = v.drop(temp_frame.index)

# Step 4
temp_frame = v[~v.Predicted_Label.isin(training)]
to_manual = to_manual.append(temp_frame, ignore_index = True)
v = v.drop(temp_frame.index)

# Save the Send to manual file
to_manual.to_csv('Images_to_manual.csv', sep = ';', index = False)
