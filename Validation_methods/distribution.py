# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:00:12 2021

@author: bpe043
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sqlite3



def calculate_log(value, total):
    
    val = value / total
    res = math.log(val)
    return res

def plot_training_production(training, production, kind, version = None, threshold = None):
    
    train_total = training['Count_training'].sum()
    production_total = production['Count_production'].sum()
    
    # log() based
    training['Fraction'] = training.apply(lambda row: calculate_log(row.Count_training, train_total), axis = 1)
    production['Fraction'] = production.apply(lambda row: calculate_log(row.Count_production, production_total), axis = 1)
        
    # Since some classes are just not represented in the production set, we add in dummy variables for them with a count and fraction of 0
    zero_values = 0
    check = production['Unique labels'].to_list()
    for index, row in training.iterrows():
        code = row['Unique labels']
        
        if code in check:
            next
        else:
            vals = math.log(1/(production_total + 2))
            insert = [code, 0, vals]
            insert = pd.DataFrame([insert], columns = ['Unique labels', 'Count_production', 'Fraction'])
            production = production.append(insert, ignore_index = True)
            zero_values += 1
            
    # We can now, after doing CTC predictions, have non-valid codes. We need to remove them from our production set
    non_valid_values = 0
    invalid_frame = pd.DataFrame(columns = ['Unique labels', 'Count_production', 'Fraction'])
    for code in check:
        if code in training['Unique labels'].values:
            next
        else:
            non_valid_values += 1
            invalid_index = production.loc[np.where(production['Unique labels'] == code)].index
            invalid_frame = invalid_frame.append(production.loc[invalid_index])
            production.drop(invalid_index, inplace = True)
            production = production.reset_index(drop = True)
            
    invalid_frame = invalid_frame.reset_index(drop = True)
            
    if kind == 'total':
        scatter(training, production, threshold)
    elif kind == 'multiple':
        scatter_multiple(training, production, threshold)
        


def scatter(training, production, threshold = None):
    
    # Pandas plotting
    plot_frame = pd.DataFrame()
    plot_frame['Fractions_training'] = training['Fraction']
    plot_frame['Fractions_production'] = production['Fraction']
    plot_frame['Codes'] = training['Unique labels']
    
    zero_value = plot_frame['Fractions_production'].loc[len(plot_frame) - 1]
    colors = ['Blue' if value == zero_value else 'Red' for value in plot_frame.Fractions_production]
    
    ax = plot_frame.plot.scatter(x = 'Fractions_training',
                                  y = 'Fractions_production',
                                  s = 4,
                                  c = colors,
                                  figsize = (10, 5))
    
    
    lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]), # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]), # max of both axes
            ]
    
    ax.plot(lims, lims, 'k-', alpha = 0.75, zorder = 0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    
    ax.set_xlabel('log(class frequency) for trainingset')
    ax.set_ylabel('log(class frequency) for full datatset')
    
    if threshold is None:
        plt.savefig('Scatterplots//scatterplot_full.png', dpi = 300)
    else:
        ax.set_title("Class distribution with a threshold of {}%".format(threshold))
        plt.savefig('Scatterplots_prod//{}//scatterplot_full.png'.format(threshold), dpi = 300)
    
    plt.show(ax)
    

def scatter_multiple(training, production, threshold):
    
    total_length = len(training)
    batch_size = 10
    batch = 0
    
    batch_start = 0
    batch_end = batch_size
    
    limit = math.ceil(total_length / batch_size)
    
    while batch < limit:
        
        plot_frame = pd.DataFrame()
        plot_frame['Fractions_training'] = training['Fraction'].iloc[batch_start:batch_end]
        plot_frame['Fractions_production'] = production['Fraction'].iloc[batch_start:batch_end]
        plot_frame['Codes'] = training['Unique labels']
        plot_frame.reset_index(drop = True, inplace = True)
        
        fig, ax = plt.subplots()
        
        colormap = plt.cm.tab10
        colorlist = [colors.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, len(plot_frame['Codes']))]
        
        # If there were no instances of the code in the production set, we color it a deep blue to make it destinct
        prod_colors = production['Count_production'].iloc[batch_start:batch_end].tolist()
        
        avg_counter = sum(prod_colors) / len(prod_colors)
        
        if 0 in prod_colors:
            for j, count in enumerate(prod_colors):
                if count == 0:
                    colorlist[j] = 'Blue'
        
        for i, c in enumerate(colorlist):
            
            x = plot_frame['Fractions_training'][i]
            y = plot_frame['Fractions_production'][i]
            l = plot_frame['Codes'][i]
            
            ax.scatter(x, y, label = l, s = 4, linewidth = 0.1, c = c)
            
        # Shrink plot to make room for legend outside of the plot
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
        ax.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        
        lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]), # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]), # max of both axes
                ]
    
        ax.plot(lims, lims, 'k-', alpha = 0.75, zorder = 0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        
        ax.set_xlabel('log(class frequency) for trainingset')
        ax.set_ylabel('log(class frequency) for full datatset')
        ax.set_title('Threshold of {}%, Batch {} \n Average predictions = {}'.format(threshold, batch, avg_counter))
        
        plt.savefig('Scatterplots_prod//{}//batch_{}.png'.format(threshold, batch), dpi = 300)
        plt.show(ax)
        
        batch += 1
        batch_start += batch_size
        batch_end += batch_size
# =============================================================================
#         # To handle the uneven length
#         if batch_end == 260:
#             batch_end += 4
#         else :
#             batch_end += batch_size
# =============================================================================
        
# We could do many things with regards to thresholding this frame, but for now it's kept simple.
def simplified_thresholds(frame, t):
    
    threshold_level = t / 100
    
    temp_frame = frame[['C0', 'C1', 'C2']]
    
    above = (temp_frame >= threshold_level).all(1)
    #below = ~(frame >= threshold_level).all(1)
    
    return frame.loc[above]
        
        
training_path = '\\\\129.242.140.132\\remote\\UtklippsDatabaser\\dugnads_sett_no_u.db'
trainingset_con = sqlite3.connect(training_path)
trainingset_c = trainingset_con.cursor()
trainingset = trainingset_c.execute("SELECT NAME, CODE FROM CELLS").fetchall()
trainingset_codes = [x[1] for x in trainingset] 
trainingset_names = [x[0] for x in trainingset]

training_data = pd.DataFrame(index = range(len(trainingset_codes)), columns = ['Labels', 'Names'])
training_data['Labels'] = trainingset_codes
training_data['Names'] = trainingset_names


# Data from predictions on our entire dataset, or our validation set from training
validationset = pd.read_csv('C:\\New_production_results\\CTC_dugnad\\total_confidence_scores.csv', sep = ';')

# Actual code
validationset = validationset.drop(validationset.columns[[0]], axis = 1)

# For testing the training resuts
# Keep only images with Correct predicted label
#validationset = validationset.iloc[np.where(validationset['Actual_Label'] == validationset['Predicted_Label'])].astype(str)

# Keep only images with Incorrect predicted label
#validationset = validationset.iloc[np.where(validationset['Actual_Label'] != validationset['Predicted_Label'])].astype(str)


# Weed out the validation images from the full trainingset based on indexes found when comparing image names
# =============================================================================
# indexes = []
# for n in validationset.Image_name:
#     training_index = training_data.loc[np.where(training_data['Names'] == n)].index.values[0]
#     indexes.append(training_index)
#         
# training_data = training_data.drop(indexes)
# training_data.reset_index(drop = True, inplace = True)
# =============================================================================
        
train_series = pd.Series(training_data['Labels'])
train_dist = train_series.value_counts().rename_axis('Unique labels').reset_index(name = 'Count_training')     
train_dist['Unique labels'].loc[4] = 'bbb'
train_dist['Unique labels'].loc[8] = 'ttt'
train_dist['Unique labels'].loc[279] = 'uuu'
                
prod_series = pd.Series(validationset['Predicted_Label'])
prod_dist = prod_series.value_counts().rename_axis('Unique labels').reset_index(name = 'Count_production')

# Since we have some invalid codes, we can use this frame instead of prod_dist
prod_valid_dist = pd.DataFrame(columns = ['Unique labels', 'Count_production'])
for index, row in train_dist.iterrows():
    
    prod_count = prod_dist['Count_production'].loc[np.where(prod_dist['Unique labels'] == row['Unique labels'])]
    
    if len(prod_count) > 0:
        prod_count = prod_count.values[0]
    
        prod_valid_dist.loc[index] = [row['Unique labels'], prod_count]



# Plot distribution With Thresholding
thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

# =============================================================================
# import os
# for i in thresholds:
#     s = str(i)
# 
#     base_path = "Scatterplots_prod//"
#     path = base_path + s
# 
#     os.mkdir(path)
# 
# =============================================================================
for t in thresholds:
    
    # Get only the images who's confidence scores all are higher than the threshold
    production_set = simplified_thresholds(validationset, t)
    prod_series = pd.Series(production_set['Predicted_Label'])
    prod_dist = prod_series.value_counts().rename_axis('Unique labels').reset_index(name = 'Count_production')
    
    # Plot
    plot_training_production(train_dist, prod_dist, 'multiple', threshold = t)
    plot_training_production(train_dist, prod_dist, 'total', threshold = t)


        
        
        
        
        
        
        
        
  
        
