# -*- coding: utf-8 -*-
"""
Created on Tue Dec  17 15:13:31 2019

@author: bpe043
"""

def get_skeleton(skeleton_file):
    with open(skeleton_file, 'r') as file:
        all_info = file.readlines()
		
    all_info.pop(0)
    skeleton = list(dict.fromkeys((item.split('<')[0],item.split('<')[1]) for item in all_info))
    del all_info
    
    return skeleton


# Finds the image in the "skeleton" file, an index of where each image is stored
def get_bone_from_skeleton(img_name, skeleton):
    
    bone = None
    
    # Find the original census image where this cell image was cut out
    for bones in skeleton:
        if img_name in bones:
            bone = bones[0] + '\\' + bones[1] + '.jpg'
            break
        
    return bone
