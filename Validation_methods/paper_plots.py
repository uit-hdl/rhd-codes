# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:26:13 2021

@author: bpe043
"""

""" Script to plot all figures found in the paper, broken down into sections """

import pandas as pd
import numpy as np
import sqlite3
import math
import matplotlib.pyplot as plt
import os

# Colors that the journal allows
Color_space = {'Pink': [241, 100, 182], 'Orange' : [245, 116, 33], 'Grey' : [209, 210, 212], 'Blue' : [20, 137, 216], 
                'Red' : [173, 57, 10], 'Sand' : [190, 152, 107], 'Grey2' : [147, 148, 152], 'Green' : [26, 68, 22]}


# Create folder to store plots
if not os.path.exists('paper_plots'):
    os.mkdir('paper_plots')


""" Helper Functions """

""" 
Function that takes in each data set in the dataframe.
The sets consist of % values. We know the total, in this case each set was made for 3.000 images.
Convert the percentage values into real values for the 'Send to Manual' column, then the other columns are percentage values based off of these (now real) values.
Meaning, if the 'Send to Manual' is 13%, that means 13% of 3.000. Then if 'Correct' is 50% and 'Incorrect' is 50%, those are 50% of the 13% that was sent to manual.
"""
def clean(frame):
    
    total = 3000
    
    for index, row in frame.iterrows():
        t = row['Threshold level']
        send = row['Send to manual']
        correct = row['Correct labels in send to manual']
        incorrect = row['Incorrect labels in send to manual']
        
        # Remove the % and convert to numerical
        t = int(t[:-1])
        send = float(send[:-1])
        correct = float(correct[:-1])
        incorrect = float(incorrect[:-1])
        
        send_real = (send * total) / 100
        correct_real = (correct * send_real) / 100
        incorrect_real = (incorrect * send_real) / 100
        
        # Update values
        frame.at[index, 'Threshold level'] = t
        frame.at[index, 'Send to manual'] = send_real
        frame.at[index, 'Correct labels in send to manual'] = correct_real
        frame.at[index, 'Incorrect labels in send to manual'] = incorrect_real
        
        
    frame = frame.astype(float)
    
    return frame
    
"""
Function that calculates the log values for the distribution of classes in the training and production sets.
Used to calculate the distribution of the two sets against each other.
"""
def calculate_log(value, total):
    
    val = value / total
    res = math.log(val)
    return res


""" 
    #####################
    # Error analysis.
    # Produces figures: 
    #####################    
"""

def total_error_per_propotion(scores, N):
    
    
    y = (scores["Keep_wrong"] + 
        0.03*(scores["Manual_correct"] + scores["Manual_wrong"]))/N
         
    x = (scores["Manual_correct"] + scores["Manual_wrong"])/N
    
    fig, ax = plt.subplots()
    
    color = np.array(Color_space['Blue'])

    ax.plot(x, y, c = color/255, zorder = 1)
    ax.scatter(x, y, c = color/255, s = 20, zorder = 2)
    
    # Remove top and right axis lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set the start/stop of the x/y axis lines
    ax.spines['bottom'].set_bounds(0, 0.12) 
    ax.spines['left'].set_bounds(0.01, 0.05) 
    
    # adjust tick spacing
    x_ticks = [x/100.0 for x in range(0, (12 + 1), 2)] 
    ax.xaxis.set_ticks(x_ticks)
    x_labels = [str(int(x*100)) + '%' for x in x_ticks]
    ax.set_xticklabels(x_labels)
    
    y_ticks = [0.01, 0.02, 0.03, 0.04, 0.05]
    ax.yaxis.set_ticks(y_ticks)
    
    y_tick_lablels = [str(int(y*100)) + '%' for y in y_ticks]
    ax.set_yticklabels(y_tick_lablels)
    
    ax.set_xlabel("Proportion of errors sent to manual validation and correction", labelpad = 10)
    ax.set_ylabel("Total error", labelpad = 10)
    
    #plt.show()
    plt.savefig('paper_plots\\total_error_per_proportion_sent_to_manual.png', dpi = 300)
    
    
    
# =============================================================================
# def derivate_error(scores, N):
#     
#     y = (scores["Keep_wrong"] + 
#     0.03*(scores["Manual_correct"] + scores["Manual_wrong"]))/N
#          
#     x = (scores["Manual_correct"] + scores["Manual_wrong"])/N
#     
#     
#     dy = [y[i] - y[i-1] for i in range(1, len(y))]
#     dx = [x[i] - x[i-1] for i in range(1, len(x))]
#     dydx = [dy[i]/dx[i] for i in range(len(dx))]
#     
#     color = np.array(Color_space['Blue'])
#     
#     fig, ax = plt.subplots()
#     
#     ax.plot(x[1:], dydx, c = color/255, linestyle = '-', zorder = 0) 
#     
#     ax.scatter(x[1:], dydx, c = color/255, s=20, zorder=1)
# 
#     # Remove top and right axis lines
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     
#     # Set the start/stop of the x/y axis lines
#     ax.spines['bottom'].set_bounds(0, 0.12) 
#     ax.spines['left'].set_bounds(-0.1, -0.8) 
#     
#     ax.set_xlabel("Proportion of errors sent to manual validation and correction", labelpad = 10)
#     ax.set_ylabel("Change in error (dy/dx)", labelpad = 10)
#     
#     plt.xlim(-0.01, 0.12)
#     plt.ylim(-0.85, -0.1)
#     
#     #plt.show()
#     plt.savefig('paper_plots\\derivative_change_in_error.png', dpi = 300)
# =============================================================================
    
    
def send_to_manual_per_threshold(scores, N):
    
    x = scores["Threshold level"]/100
    y = (scores["Manual_correct"] + scores["Manual_wrong"])/N  
    
    fig, ax = plt.subplots()
    
    color = np.array(Color_space['Blue'])
    
    # Plotting the Accuracy (Of the X images that passed each threshold, how many of them were Correct) as a second y-axis
    scores['Passed_threshold'] = scores['Keep_correct'] + scores['Keep_wrong']
    y2 = (scores['Keep_correct'] / scores['Passed_threshold']) * 100
    color2 = np.array(Color_space['Orange'])
    ax2 = ax.twinx()
    
    
    line1 = ax.plot(x, y, linestyle = '-', linewidth = 1.5, c = color/255, zorder = 1, label = 'Proportion sent to manual')
    
    ax.scatter(x, y, c = color/255, s = 20, zorder = 3)
    
    ci = 1.6 * math.sqrt(np.mean(y) * (1-np.mean(y)) / N)
    ax.fill_between(x, (y-ci), (y+ci), color = np.array(Color_space['Grey'])/255, alpha = .4)
    ax.text(0.85, 0.25, '90% interval', horizontalalignment = 'center', verticalalignment = 'center', transform = ax.transAxes, color = np.array(Color_space['Grey'])/255, alpha = .8)
    
    ax.set_xlabel("Confidence threshold", labelpad = 10)
    ax.set_ylabel("Proportion sent to manual", labelpad = 10)
    
    line2 = ax2.plot(x, y2, linestyle = '-', linewidth = 1.5, c = color2/255, zorder = 1, label = 'Accuracy per threshold')
    ax2.scatter(x, y2, c = color2/255, s = 20, zorder = 3)
    
    ax2.set_ylabel('Accuracy per threshold', labelpad = 10)
    
    ax.set_xlim([-.05, ax.get_xlim()[1] + 0.05])
    ax.set_ylim([-.01, ax.get_ylim()[1]])
    ax2.set_ylim([95, 100])
    
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    ax.spines['bottom'].set_bounds(.05, .95)
    ax2.spines['bottom'].set_bounds(.05, .95)
    
    ax.spines['left'].set_bounds(0, 0.12) 
    ax2.spines['left'].set_bounds(96, 100) 
    
    y_ticks_1 = [y/100.0 for y in range(0, 12 + 1, 2)]
    ax.yaxis.set_ticks(y_ticks_1)
    
    x_ticks = [x/100.0 for x in range(5, (95 + 1), 15)] 
    ax.xaxis.set_ticks(x_ticks)
    
    y1_tick_labels = [str(int(y*100)) + '%' for y in y_ticks_1]
    ax.set_yticklabels(y1_tick_labels)
    
    y2_ticks = [95, 96, 97, 98, 99, 100]
    y2_tick_lablels = [str(y) + '%' for y in y2_ticks]
    ax2.set_yticklabels(y2_tick_lablels)
    
    # Legend
    labels = line1 + line2
    legends = [l.get_label() for l in labels]
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(labels, legends, loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon = False, ncol=2)
    
    
    plt.tight_layout()
    #plt.show()
    plt.savefig('paper_plots\\manual_accuracy_perThreshold.png', dpi = 300, bbox_inches = 'tight')
    
    
def sentToManual_CorrectVsIncorrect():
    
    # v here is the thresholds csv file generated by thresholds.py for the training results.
    v = pd.read_csv("C:\\New_production_results\\CTC_dugnad\\production_example_thresholds.csv", sep = ';')
    v = v[['Threshold level', 'Send to manual', 'Correct labels in send to manual', 'Incorrect labels in send to manual']]
    
    # Clean and reformat
    v = clean(v)
    v['Threshold level'] = v['Threshold level']
    
    colors = [np.array(Color_space['Blue'])/255, np.array(Color_space['Orange'])/255]
    x = v['Threshold level'].tolist()
    y = [v['Correct labels in send to manual'].tolist(), v['Incorrect labels in send to manual'].tolist()]
    
    fig, ax = plt.subplots()
    
    ax.stackplot(x, y, labels = ['Correct images sent to manual', 'Incorrect images sent to manual'], colors = colors)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.spines['bottom'].set_bounds(5, 95)
    ax.spines['left'].set_bounds(0, 300) 
    
    
    x_ticks = [x for x in range(5, (95 + 1), 15)]
    ax.xaxis.set_ticks(x_ticks)
    x_tick_labels = [x/100 for x in x_ticks]
    ax.set_xticklabels(x_tick_labels)
    
    ax.set_xlabel("Confidence threshold", labelpad = 10)
    ax.set_ylabel("Images", labelpad = 10)
    
    # Move legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), frameon = False, ncol=2)
    
    #plt.show()
    fig.savefig('paper_plots\\CorrectVsIncorrect_sentToManual.png', dpi = 300, bbox_inches = 'tight')
        
    
    
""" 
    #####################
    # Distribution analysis.
    # Produces figures: 
    #####################    
"""
    
def old_distribution(old_trainingset_path, old_results_path):
    
    # Path to training database
    training_path = old_trainingset_path
    db_train = sqlite3.connect(training_path)
    trainingset = db_train.cursor().execute("SELECT Code from cells").fetchall()
    trainingset = [train for t in trainingset for train in t]
    train_series = pd.Series(trainingset)
    train_dist = train_series.value_counts().rename_axis('Unique labels').reset_index(name = 'Count_training')
    
    # (OLD) Results from model
    full_set = pd.read_csv(old_results_path, sep = ',')
    full_set_labels = full_set['Predicted label']
    prod_dist = full_set_labels.value_counts().rename_axis('Unique labels').reset_index(name = 'Count_production')
    prod_dist['Unique labels'] = prod_dist['Unique labels'].astype(str)
    
    train_total = train_dist['Count_training'].sum()
    production_total = prod_dist['Count_production'].sum()
    
    # log() based
    train_dist['Fraction'] = train_dist.apply(lambda row: calculate_log(row.Count_training, train_total), axis = 1)
    prod_dist['Fraction'] = prod_dist.apply(lambda row: calculate_log(row.Count_production, production_total), axis = 1)
    
    
    
    # Since some classes are just not represented in the production set, we add in dummy variables for them with a count and fraction of 0
    check = prod_dist['Unique labels'].to_list()
    for index, row in train_dist.iterrows():
        code = row['Unique labels']
        
        if code in check:
            next
        else:
            vals = math.log(1/(production_total + 2))
            insert = [code, 0, vals]
            insert = pd.DataFrame([insert], columns = ['Unique labels', 'Count_production', 'Fraction'])
            prod_dist = prod_dist.append(insert, ignore_index = True)
            
            
    # Pandas plotting
    plot_frame = pd.DataFrame()
    plot_frame['Fractions_training'] = train_dist['Fraction']
    plot_frame['Fractions_production'] = prod_dist['Fraction']
    plot_frame['Codes'] = train_dist['Unique labels']
    
    ax = plot_frame.plot.scatter(x = 'Fractions_training',
                                  y = 'Fractions_production',
                                  s = 4,
                                  c = np.array(Color_space['Blue'])/255
                                  )
    
    lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]), # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]), # max of both axes
            ]
    
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # Just for stylistic reasons
    lims[0] = lims[0] + 1 
    ax.plot(lims, lims, 'k-', alpha = 0.75, zorder = 0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
        
    ax.spines['bottom'].set_bounds(0, -14)
    ax.spines['left'].set_bounds(0, -14) 
    
    ax.set_xlabel('log(class frequency) for training', labelpad = 10)
    ax.set_ylabel('log(class frequency) for results', labelpad = 10)
    
    ticks = [x for x in range(-14, (0+1), 2)]
        
    ax.xaxis.set_ticks(ticks)
    ax.yaxis.set_ticks(ticks)
    
    plt.tight_layout()
    plt.savefig('paper_plots\\less_good_distribution.png', dpi = 600)
    
    
def new_distribution(trainingset_path, results):
    
    # Path to training database
    training_path = trainingset_path
    db_train = sqlite3.connect(training_path)
    trainingset = db_train.cursor().execute("SELECT Code from cells").fetchall()
    trainingset = [train for t in trainingset for train in t]
    train_series = pd.Series(trainingset)
    train_dist = train_series.value_counts().rename_axis('Unique labels').reset_index(name = 'Count_training')
    
    # (NEW) Results from model
    full_set = pd.read_csv(results, sep = ';')
    full_set_labels = full_set['Predicted_Label']
    prod_dist = full_set_labels.value_counts().rename_axis('Unique labels').reset_index(name = 'Count_production')
    prod_dist['Unique labels'] = prod_dist['Unique labels'].astype(str)
    
    # We can now, after doing CTC predictions, have non-valid codes. We need to remove them from our production set
    check = prod_dist['Unique labels'].to_list()
    
    non_valid_values = 0
    for code in check:
        if code in train_dist['Unique labels'].values:
            next
        else:
            non_valid_values += 1
            invalid_index = prod_dist.loc[np.where(prod_dist['Unique labels'] == code)].index
            prod_dist.drop(invalid_index, inplace = True)
            prod_dist = prod_dist.reset_index(drop = True)
            
    
    train_total = train_dist['Count_training'].sum()
    production_total = prod_dist['Count_production'].sum()
    
    # log() based
    train_dist['Fraction'] = train_dist.apply(lambda row: calculate_log(row.Count_training, train_total), axis = 1)
    prod_dist['Fraction'] = prod_dist.apply(lambda row: calculate_log(row.Count_production, production_total), axis = 1)
    
    
    
    # Since some classes are just not represented in the production set, we add in dummy variables for them with a count and fraction of 0
    check = prod_dist['Unique labels'].to_list()
    for index, row in train_dist.iterrows():
        code = row['Unique labels']
        
        if code in check:
            next
        else:
            vals = math.log(1/(production_total + 2))
            insert = [code, 0, vals]
            insert = pd.DataFrame([insert], columns = ['Unique labels', 'Count_production', 'Fraction'])
            prod_dist = prod_dist.append(insert, ignore_index = True)
            
            
    # Pandas plotting
    plot_frame = pd.DataFrame()
    plot_frame['Fractions_training'] = train_dist['Fraction']
    plot_frame['Fractions_production'] = prod_dist['Fraction']
    plot_frame['Codes'] = train_dist['Unique labels']
    
    ax = plot_frame.plot.scatter(x = 'Fractions_training',
                                  y = 'Fractions_production',
                                  s = 4,
                                  c = np.array(Color_space['Blue'])/255
                                  )
    
    lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]), # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]), # max of both axes
            ]
    
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # Just for stylistic reasons
    lims[0] = lims[0] + 1
    ax.plot(lims, lims, 'k-', alpha = 0.75, zorder = 0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.spines['bottom'].set_bounds(0, -15)
    ax.spines['left'].set_bounds(0, -15) 
    
    ticks = [x for x in range(-15, (0+1), 3)]
        
    ax.xaxis.set_ticks(ticks)
    ax.yaxis.set_ticks(ticks)
    
    ax.set_xlabel('log(class frequency) for training', labelpad = 10)
    ax.set_ylabel('log(class frequency) for results', labelpad = 10)
    
    plt.tight_layout()
    plt.savefig('paper_plots\\better_distribution.png', dpi = 600)
    

def character_frequency(trainingset_path):
    
    # Training set
    training = sqlite3.connect(trainingset_path)
    codes = pd.read_sql("SELECT CODE FROM CELLS", training)
    codes = codes.Code.tolist()

    total = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, 'b': 0, 't': 0} 
    
    """ For an even more in-depth view """
# =============================================================================
#     firsts = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, 'b': 0, 't': 0}
#     seconds = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, 'b': 0, 't': 0}
#     thirds = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, 'b': 0, 't': 0}
# =============================================================================
    
    for x in codes:
        
        if len(x) == 1:
            x = x*3
        
        f = x[0]
        s = x[1]
        t = x[2]
        
        total[f] += 1
        total[s] += 1
        total[t] += 1
        
# =============================================================================
#         firsts[f] += 1
#         seconds[s] += 1
#         thirds[t] += 1
# =============================================================================


    fig, ax = plt.subplots()
    ax.bar(list(total.keys()), total.values(), color = np.array(Color_space['Blue'])/255, width = 0.6)
    
           
    # Remove axis lines
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Remove tick marks
    ax.tick_params(bottom = False, left = False,)
    
    # Add bar lines as a horizontal grid
    ax.yaxis.grid(color = "white" )
    
    ax.set_ylabel('Frequency', labelpad = 10)
    ax.set_xlabel('Characters', labelpad = 10)
    
    plt.tight_layout()
    plt.savefig('paper_plots\\char_frequency.png', dpi = 300)


def cumulative_dist(old_trainingset_path):
    
    # Path to (OLD) training database
    training_path = old_trainingset_path
    db_train = sqlite3.connect(training_path)
    trainingset = db_train.cursor().execute("SELECT Code from cells").fetchall()
    trainingset = [train for t in trainingset for train in t]
    train_series = pd.Series(trainingset)    
    training_count = pd.DataFrame(train_series, columns = ['Codes'])

    
    # First frame, training set
    stats_df = training_count \
    .groupby('Codes') \
    ['Codes'] \
    .agg('count') \
    .pipe(pd.DataFrame) \
    .rename(columns = {'Codes' : 'frequency'})
    
    # PDF
    stats_df['pdf'] = stats_df['frequency'] / sum(stats_df['frequency'])
    
    # CDF
    stats_df = stats_df.sort_values(by=['pdf'], ascending = False)
    stats_df = stats_df['pdf'].multiply(other = 100).reset_index(drop=True).reset_index()
    stats_df['cdf training'] = stats_df['pdf'].cumsum()
    
    color = np.array(Color_space['Blue'])/255
    ax = stats_df.plot(x = 'index', y = ['cdf training'], legend = False, c = color)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
    ax.spines['bottom'].set_bounds(0, 264) 
    ax.spines['left'].set_bounds(0, 100) 
    
    x_ticks = [x for x in range(0, (264 + 1), 33)]
    ax.xaxis.set_ticks(x_ticks)
    
    y_ticks = [y for y in range(0, (100 + 1), 20)]
    ax.yaxis.set_ticks(y_ticks)
    
    ax.set_xlim([- 10, 270])
    ax.set_ylim([- 10, 105])
    
    # Dots
    indexes = [0, 33, 66, 99, 132, 165, 198, 231, 263]
    values = stats_df['cdf training'].iloc[indexes]
    ax.scatter(x_ticks, values, c = color, s = 20, zorder = 2)
    
    ax.set_xlabel("Codes", labelpad = 10)
    ax.set_ylabel("Cumulative distribution", labelpad = 10)
    
    plt.tight_layout()
    plt.savefig('paper_plots\\cumulative_distribution.png', dpi = 300)
    #plt.show()
    
    

# Plotting
#sentToManual_CorrectVsIncorrect()
    
# Threshold values from the Verification dataset that were given as output from the model
path = "C://New_production_results//CTC_dugnad//production_example_thresholds_real.csv"
scores = round(pd.read_csv(path, delimiter=";"))
N = sum(scores.iloc[1, 1:]) # sum of counts along one row is total number of obs.

# =============================================================================
# # Plots using those data sources #
# total_error_per_propotion(scores, N)
# derivate_error(scores, N)
# send_to_manual_per_threshold(scores, N)
# 
# 
# # The path to the Dugnad training set database
# trainingset_path = '\\\\129.242.140.132\\remote\\UtklippsDatabaser\\dugnads_sett_no_u.db'
# 
# # The path to the full census results
# results = "C://New_production_results//CTC_dugnad//total_confidence_scores.csv"
# 
# # Plots using that data source #
# character_frequency(trainingset_path)
# new_distribution(trainingset_path, results)
# 
# 
# # The path to the old results from the 3-digit model
# old_results_path = 'C:\\New_production_results\\11_2020\\All_results.csv'
# # The path to the old training set (2% set)
# old_trainingset_path = '\\\\129.242.140.132\\remote\\UtklippsDatabaser\\full_3digit_trainingset.db'
# 
# # Plots using those data sources #
# cumulative_dist(old_trainingset_path)
# old_distribution(old_trainingset_path, old_results_path)
# 
# =============================================================================




total_error_per_propotion(scores, N)























