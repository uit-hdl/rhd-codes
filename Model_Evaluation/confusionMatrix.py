# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 08:46:44 2019

@author: bpe043

Pretty print code from Zachguo
https://gist.github.com/zachguo/10296432

Added code to make it print to file


"""

import pickle
import os
from sklearn.metrics import confusion_matrix

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    
    with open('confusion_matrix.txt', 'a') as f:
        
        columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
        empty_cell = " " * columnwidth
        # Print header
        f.write("    " + empty_cell)
        for label in labels:
            f.write("%{0}s".format(columnwidth) % label)
        f.write('\n')
        # Print rows
        for i, label1 in enumerate(labels):
            f.write("    %{0}s".format(columnwidth) % label1)
            for j in range(len(labels)):
                cell = "%{0}.0f".format(columnwidth) % cm[i, j]
                if hide_zeroes:
                    cell = cell if float(cm[i, j]) != 0 else empty_cell
                if hide_diagonal:
                    cell = cell if i != j else empty_cell
                if hide_threshold:
                    cell = cell if cm[i, j] > hide_threshold else empty_cell
                f.write(cell)
            f.write('\n')

# Load in the lists used to calculate the score
if os.path.exists('labels.txt'):
    with open('labels.txt', 'rb') as fp:
        list_of_labels = pickle.load(fp)
        del list_of_labels[0]

if os.path.exists('predictions.txt'):
    with open('predictions.txt', 'rb') as fp:
        list_of_predictions = pickle.load(fp)
        
labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        
cf = confusion_matrix(list_of_labels, list_of_predictions)

print_cm(cf, labels)
