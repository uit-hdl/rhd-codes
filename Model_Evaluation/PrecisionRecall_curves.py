# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:01:04 2019

@author: bpe043
"""

from sklearn.utils import assert_all_finite
from sklearn.utils.extmath import stable_cumsum

import warnings
import numbers


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
#from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize

from sklearn.metrics import roc_curve, auc
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

if os.path.exists('labels.txt'):
    with open('labels.txt', 'rb') as fp:
        list_of_labels = pickle.load(fp)

if os.path.exists('predictions.txt'):
    with open('predictions.txt', 'rb') as fp:
        list_of_predictions = pickle.load(fp)
        
if os.path.exists('prediction_values.txt'):
    with open('prediction_values.txt', 'rb') as fp:
        list_of_prediction_values = pickle.load(fp)
        
if os.path.exists('full_prediction_values.txt'):
    with open('full_prediction_values.txt', 'rb') as fp:
        full_prediction_values = pickle.load(fp)
# =============================================================================
# target_names = ['Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5', 'Label 6', 'Label 7', 'Label 8', 'Label 9']
# 
# cr = classification_report(list_of_labels, list_of_predictions, target_names = target_names)
# 
# print(cr)
# =============================================================================
          
start = 0
step = 100
i = 1               # Used as the pos_label parameter for precision_recall_curve, hence starting at 1
while i < 10:
     
    predictions = list_of_predictions[start: start + step]
    prediction_values = list_of_prediction_values[start : start + step]
    
    predictions = np.array(predictions)
    
    precision, recall, thresholds = precision_recall_curve(predictions, prediction_values, pos_label = i)

    plt.plot(thresholds, precision, color=sns.color_palette()[1])
    plt.plot(thresholds, recall, color=sns.color_palette()[0])

    
    leg = plt.legend(('precision', 'recal'), frameon=True, loc='lower left')
    leg.get_frame().set_edgecolor('k')
    plt.title('Precision - Recall for label {}'.format(i))
    plt.xlabel('threshold')
    plt.ylabel('%')

    
    # Extra visibility factors
# =============================================================================
#     plt.xticks([0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0])
#     plt.yticks([0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0], fontsize = 6)
#     plt.grid()
#     plt.axis([0.8, 1.05, 0.8, 1.025])
# =============================================================================

    plt.savefig('Plots/Precision - Recall for label {}.png'.format(i))
    
    plt.show()
        
    stop = True
    i += 1
    start += 100


