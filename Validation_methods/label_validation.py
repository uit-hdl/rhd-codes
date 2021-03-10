# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:50:25 2021

@author: bpe043
"""

import numpy as np
import pandas as pd
import sqlite3


training_path = '\\\\129.242.140.132\\remote\\UtklippsDatabaser\\dugnads_sett_no_u.db'
trainingset_con = sqlite3.connect(training_path)
trainingset_c = trainingset_con.cursor()
trainingset = trainingset_c.execute("SELECT NAME, CODE FROM CELLS").fetchall()
trainingset_codes = [x[1] for x in trainingset] 
trainingset_names = [x[0] for x in trainingset]

training_data = pd.DataFrame(index = range(len(trainingset_codes)), columns = ['Labels', 'Names'])
training_data['Labels'] = trainingset_codes
training_data['Names'] = trainingset_names

train_series = pd.Series(training_data['Labels'])
train_dist = train_series.value_counts().rename_axis('Unique labels').reset_index(name = 'Count_training')     
train_dist['Unique labels'].loc[4] = 'bbb'
train_dist['Unique labels'].loc[8] = 'ttt'
train_dist['Unique labels'].loc[279] = 'uuu'


# Data from predictions on our entire dataset, or our validation set from training
validationset = pd.read_csv('C:\\New_production_results\\CTC_dugnad\\total_confidence_scores.csv', sep = ';')
frame = validationset[['C0', 'C1', 'C2']]
non_numeric = validationset[['Image_name', 'Predicted_Label']]

total = len(validationset)

thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

results = pd.DataFrame(index = np.arange(len(thresholds)), columns = ['Total_images', 'Threshold_level', 'Keep', 'To_manual', 'Valid_classes', 'Potentially_valid_classes', 'Invalid_classes', 'Valid_images', 'Potentially_valid_images', 'Invalid_images', 'Perc_Valid_images', 'Perc_Potentially_valid_images', 'Perc_Invalid_images'])

# Adding a threshold
for t in thresholds:
    
    
    threshold_level = t / 100
    print("\n\nAt a threshold of {} we have:".format(t))
    
    above = (frame >= threshold_level).all(1)
    below = ~(frame >= threshold_level).all(1)
    
    keep_scores = frame.loc[above]
    keep_nn = non_numeric.loc[above]
    val_set = keep_scores.merge(keep_nn, left_index = True, right_index = True)
    
    send_to_manual = frame.loc[below]
    
    perc_keep = (len(keep_scores) / total) * 100
    perc_manual = (len(send_to_manual) / total) * 100
    
    prod_series = pd.Series(val_set['Predicted_Label'])
    prod_dist = prod_series.value_counts().rename_axis('Unique labels').reset_index(name = 'Count_production')
    
    
    definitely_valids = pd.DataFrame(columns = ['Unique labels', 'Count'])
    for index, row in train_dist.iterrows():
        
        prod_count = prod_dist['Count_production'].loc[np.where(prod_dist['Unique labels'] == row['Unique labels'])]
        
        if len(prod_count) > 0:
            prod_count = prod_count.values[0]
        
            definitely_valids.loc[index] = [row['Unique labels'], prod_count]
            
            
    definitely_valids_sum = sum(definitely_valids['Count'])
    print('We have {} images that are definitely valid, with {} classes!'.format(definitely_valids_sum, len(definitely_valids)))
            
    # Find all the labels that MIGHT NOT be valid
    potentially_valid = pd.DataFrame(columns = ['Unique labels', 'Count'])
    for index, row in prod_dist.iterrows():
        
        if row['Unique labels'] not in definitely_valids['Unique labels'].values:
            potentially_valid.loc[index] = [row['Unique labels'], row['Count_production']]
        
        
        
        
    #From these, we can find some labels that ARE DEFINITELY NOT VALID, for instance if they are some combination of t/b/number, or they contain the 'UNK' of an unknown prediction
    definitely_invalid = pd.DataFrame(columns = ['Unique labels', 'Count'])
    for index, row in potentially_valid.iterrows():
        
        s = row['Unique labels']
        
        has_number = False
        has_t = False
        has_b = False
        is_unk = False
        
        has_number = any(char.isdigit() for char in s)
        has_t = 't' in s
        has_b = 'b' in s
        is_unk = 'UNK' in s
        
        # Method of checking if more than one boolean var is true
        if sum(map(bool, [has_number, has_t, has_b, is_unk])) > 1:
            
            definitely_invalid.loc[index] = [row['Unique labels'], row['Count']]
            
            
    definitely_invalid_sum = sum(definitely_invalid['Count'])
    print('And of those, we have {} images that are definitely invalid, with {} classes!'.format(definitely_invalid_sum, len(definitely_invalid)))
    
    # Then we remove the definitely invalid codes from our potentially valid ones
    potentially_valid = potentially_valid.drop(definitely_invalid.index, axis = 0)
    
    potentially_valid_sum = sum(potentially_valid['Count'])
    print('We have {} images that are potentially valid, with {} classes!'.format(potentially_valid_sum, len(potentially_valid)))
    
    perc_valid = (definitely_valids_sum / total) * 100
    perc_potential = (potentially_valid_sum / total) * 100
    perc_invalid = (definitely_invalid_sum / total) * 100
            
    
    
    results.loc[thresholds.index(t)] = ['{}'.format(total), '{}%'.format(t), '{}%'.format(perc_keep), '{}%'.format(perc_manual), '{}'.format(len(definitely_valids)), '{}'.format(len(potentially_valid)), '{}'.format(len(definitely_invalid)), '{}'.format(definitely_valids_sum), '{}'.format(potentially_valid_sum), '{}'.format(definitely_invalid_sum), '{}%'.format(perc_valid), '{}%'.format(perc_potential), '{}%'.format(perc_invalid)]
    
        
    
results.to_csv('Scatterplots_prod//label_validation.csv', sep = ';', encoding = 'utf-8', index = False)