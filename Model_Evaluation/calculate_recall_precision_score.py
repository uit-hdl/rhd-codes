# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 09:32:29 2019

@author: bpe043
"""

import pickle
import os

from sklearn.metrics import precision_score, recall_score, f1_score

# Load in the lists used to calculate the score
if os.path.exists('labels.txt'):
    with open('labels.txt', 'rb') as fp:
        list_of_labels = pickle.load(fp)

if os.path.exists('predictions.txt'):
    with open('predictions.txt', 'rb') as fp:
        list_of_predictions = pickle.load(fp)

# Write precision recall score to file                
with open('score_balanced.txt', 'a') as f:
    f.write('Precision (micro): %f \n' % precision_score(list_of_labels, list_of_predictions, average = 'micro'))
    f.write('Recall (micro): %f \n' % recall_score(list_of_labels, list_of_predictions, average = 'micro'))
    f.write('F1 score (micro): %f \n' % f1_score(list_of_labels, list_of_predictions, average = 'micro'))
    f.write('\n')
    
    f.write('Precision (macro): %f \n' % precision_score(list_of_labels, list_of_predictions, average = 'macro'))
    f.write('Recall (macro): %f \n' % recall_score(list_of_labels, list_of_predictions, average = 'macro'))
    f.write('F1 score (macro): %f \n' % f1_score(list_of_labels, list_of_predictions, average = 'macro'))
    f.write('\n')
    
    f.write('Precision (weighted): %f \n' % precision_score(list_of_labels, list_of_predictions, average = 'weighted'))
    f.write('Recall (weighted): %f \n' % recall_score(list_of_labels, list_of_predictions, average = 'weighted'))
    f.write('F1 score (weighted): %f \n' % f1_score(list_of_labels, list_of_predictions, average = 'weighted'))
    f.write('\n')
    
# =============================================================================
# # Write out f1, recall, precision for each label
# precision, recall, fscore, support = score(list_of_labels, list_of_predictions)
# labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# headers = ['Label', 'Precision', 'Recall', 'FScore', 'Instances']
# 
# t = Texttable()
# 
# t.add_row(headers)
# 
# i = 0
# while i < 9:
#     t.add_row([labels[i], precision[i], recall[i], fscore[i], support[i]])
#     i += 1
#     
# print(t.draw())
# 
# with open('score_centered.txt', 'a') as f:
#     f.write(t.draw())
# =============================================================================


