# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:07:14 2021

@author: bpe043
"""

import pandas as pd
import numpy as np

import os
import glob

def merge_batch_results(results_folder_path):
    
    os.chdir(results_folder_path)
    all_filenames = [i for i in glob.glob('*.csv')]
    
    # Combine all the files in the list
    combined_frame = pd.concat([pd.read_csv(f, sep = ';') for f in all_filenames])    
    
    # Export to csv
    combined_frame.to_csv(results_folder_path + '\\total_confidence_scores.csv', sep = ';', index = False)
    
    


def thresholding(path, training = False):
    
    thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    
    cf_frame = pd.read_csv(path + '//total_confidence_scores.csv', sep = ';', encoding = "utf-8")
    cf_frame.drop(cf_frame.columns[0], axis = 1, inplace = True)
    
    if training == True:
        # For testing the training resuts when we have correct labels
        non_numeric = cf_frame[['Actual_Label', 'Predicted_Label']]
        frame = cf_frame.drop(cf_frame.columns[[0, 1]], axis = 1)
        output_frame = pd.DataFrame(index = np.arange(len(thresholds)), columns = ['Threshold level', 'Keep', 'Send to manual', 'Correct lables in keep', 'Incorrect labels in keep', 'Correct labels in send to manual', 'Incorrect labels in send to manual'])
    else:
        # For the full dataset, where we don't know what the images really contain
        non_numeric = cf_frame[['Image_name', 'Predicted_Label']]
        frame = cf_frame.drop(cf_frame.columns[[0, 1]], axis = 1)
        output_frame = pd.DataFrame(index = np.arange(len(thresholds)), columns = ['Threshold level', 'Keep', 'Send to manual'])
    
    
    
    total_length = len(frame)
    
    
    for t in thresholds:
        
        threshold_level = t / 100
        
        above = (frame >= threshold_level).all(1)
        below = ~(frame >= threshold_level).all(1)
        
        keep_scores = frame.loc[above]
        keep_nn = non_numeric.loc[above]
        keep = keep_scores.merge(keep_nn, left_index = True, right_index = True)
        
        if training == True:
            correct_labled_keep = keep.iloc[np.where(keep['Actual_Label'] == keep['Predicted_Label'])].astype(str)
            incorrect_labled_keep = keep.iloc[np.where(keep['Actual_Label'] != keep['Predicted_Label'])].astype(str)
            perc_correct_labled_keep = (len(correct_labled_keep) / len(keep)) * 100
            perc_incorrect_labled_keep = (len(incorrect_labled_keep) / len(keep)) * 100
        
        send_to_manual_scores = frame.loc[below]
        manual_nn = non_numeric.loc[below]
        send_to_manual = send_to_manual_scores.merge(manual_nn, left_index = True, right_index = True)
        
        if training == True:
            correct_labled_manual= send_to_manual.iloc[np.where(send_to_manual['Actual_Label'] == send_to_manual['Predicted_Label'])].astype(str)
            incorrect_labled_manual= send_to_manual.iloc[np.where(send_to_manual['Actual_Label'] != send_to_manual['Predicted_Label'])].astype(str)
            
            # Avoid dividing by zero
            if len(send_to_manual) > 0:
                perc_correct_labled_manual = (len(correct_labled_manual) / len(send_to_manual)) * 100
                perc_incorrect_labled_manual = (len(incorrect_labled_manual) / len(send_to_manual)) * 100
                
            else:
                perc_correct_labled_manual = 0
                perc_incorrect_labled_manual = 0
            
        number_keep = len(keep)
        number_manual = len(send_to_manual)
        perc_keep = (number_keep / total_length) * 100
        perc_manual = (number_manual / total_length) * 100
        
        if training == True:
            output_frame.loc[thresholds.index(t)] = ['{}%'.format(t), '{}%'.format(perc_keep), '{}%'.format(perc_manual), '{}%'.format(perc_correct_labled_keep), '{}%'.format(perc_incorrect_labled_keep), '{}%'.format(perc_correct_labled_manual), '{}%'.format(perc_incorrect_labled_manual)]
        else:
            output_frame.loc[thresholds.index(t)] = ['{}%'.format(t), '{}%'.format(perc_keep), '{}%'.format(perc_manual)]
    
    # Round off to 2 decimals 
    output_frame = output_frame.round(2)
    output_frame.to_csv(path + '//thresholds.csv', sep = ';', encoding = 'utf-8')
    
    
thresholding('C:\\New_production_results\\CTC_dugnad')