# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:26:03 2019

@author: bpe043
"""

from shutil import copy
import pandas as pd
import os

# Example filename Z:\bilder\BIRTH_PLACE\006KID67911_K1-8\fs10061409083841_row-6.jpg

frequence = 'Frekvensliste.txt'
images = 'Fsted.txt'

# Frekvensliste
freq = pd.read_csv(frequence, sep = ';', header=None, encoding = "ISO-8859-1" )
freq.columns = ['Count', 'Place']

freq = freq.drop(freq.index[25:], axis=0)
# =============================================================================
# freq = freq.drop(freq.index[0], axis = 0)
# freq.reset_index(drop=True, inplace=True)
# =============================================================================

birthplaces = pd.read_pickle('birthplaces.plk')


root = 'Z:/bilder/BIRTH_PLACE/'
image_root = 'BIRTH_PLACE_form-'
# Iterate over Place, select the first 1000 of the randomized values
for index, row in freq.iterrows():    
    place = row['Place']
    
    # If a folder does not exist for the place, make one
    output = 'Birthplaces/'
    
    # Crossed out data needs to be renamed from what it's called in the database
    if place == '<>':
        output += 'Strek'
    else: output += place
    
    if not os.path.exists(output):
        os.mkdir(output)
    
    temp = birthplaces[birthplaces['BP'] == place][:1000]

    # Iterate over temp now. Get Folder, Image, and Row and concatenate them. This will be the path to each picture
    for index2, row2 in temp.iterrows():
        folder = row2['Mappe']
        image = row2['OrigbildeID']
        row = str(row2['Rad'])
        
        path = root + folder + '/' + image_root + image + '_row-' + row + '.jpg' 
        
        # Copy image to directory of Place
        copy(path, output)
        
        

    
    
