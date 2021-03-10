# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:35:27 2021

@author: bpe043
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd

from datetime import datetime
import gc

def encode_single_sample_inference(img):
    # 1. Read image
    # For us, this is just the byte data from the database
    #img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    return {"image": img}



def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length = input_len, greedy = True)[0][0][:, :max_length]
    
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
        
    return output_text



# Get images from the database
def get_images(cur, start, end):
    
    data = cur.execute("SELECT Name, Image FROM FIELDS WHERE ROWID > {} AND ROWID < {}".format(start, end)).fetchall()
    
    return data
    



# Desired image dimensions
img_width = 200
img_height = 115

# For inference, the batch size does not matter
# As long as you can fit the batch in your GPU(S), it can be as high as possible. Will make inference faster
batch_size = 50000

# Maximum length of any label in the dataset
# For us, this would be 3
max_length = 3

# Our "classes"
# Defining this list and the order of the "classes" will make sure that the output is propperly mapped back
characters = ['b', '1', '0', '2', '5', '6', '7', 't', '9', '8', '4', '3', 'u']

# Mapping characters to integers
char_to_num = layers.experimental.preprocessing.StringLookup(vocabulary = list(characters), num_oov_indices=0, mask_token = None)

# Mapping integers back to original characters
num_to_char = layers.experimental.preprocessing.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)


def main(batch_index, start, end, cur, prediction_model, exclusion_set):
        
    print("Getting images from the database at {}".format(datetime.now().time()))
    
    data = get_images(cur, start, end)
    
    if len(data) == 0:
        return False
    
    print("Starting the process of excluding images from the batch at {}".format(datetime.now().time()))
    
    data_names = [x[0] for x in data]
    
    # Remove all the indexes from names_and_images where the image is in the exclusion_set
    keep_indexes = []
    keep_index = 0
    for n in data_names:
        if n not in exclusion_set:
            keep_indexes.append(keep_index)
        
        keep_index += 1
        
    
    data = [data[i] for i in keep_indexes]

    
    print("Starting prediction at {}".format(datetime.now().time()))
    
    img_bytes = [x[1] for x in data]
    names = [x[0] for x in data]


    validation_dataset = tf.data.Dataset.from_tensor_slices(img_bytes)
    validation_dataset = (
        validation_dataset.map(
            encode_single_sample_inference, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )
        

    # Our batch size is equal to the amount of images we load in, so we only need to "take" 1
    for batch in validation_dataset.take(1):
        batch_images = batch['image']
        
        preds = prediction_model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds)
        
    
    print("Starting to assign confidence scores at {}".format(datetime.now().time()))
    
    # Create a dataframe to contain the prediction and confidence score for each image in the current batch
    confidence_scores_batch = pd.DataFrame(index = np.arange(len(pred_texts)), columns = ['Image_name', 'Predicted_Label', 'C0', 'C1', 'C2'])
    
    # Add data to our dataframe
    for i in range(len(pred_texts)):
        
        conf_start = preds[i].max(axis = 1)
        lab_start = np.argmax(preds[i], axis = 1)

        
        # Remove the confidences that were put as "Blank"
        confs = np.delete(conf_start, np.where(lab_start == 13))
        
        # If we don't get 3 confidence scores back, we can't prooperly classify the image. We set all confidence scores to be 0
        if len(confs) != 3:
            confs = (0, 0, 0)
        
        confidence_scores_batch.loc[i] = [names[i], pred_texts[i], float(confs[0]), float(confs[1]), float(confs[2])]
        
    print("Starting to store confidence scores at {}".format(datetime.now().time()))
        
    # Store the confidence score dataframe
    confidence_scores_batch = confidence_scores_batch.round(2)
    confidence_scores_batch.to_csv('C://New_production_results//CTC_dugnad//Batch_{}.csv'.format(batch_index), sep = ';', encoding = 'utf-8')
    
    del data
    del keep_indexes
    del img_bytes
    del names
    del validation_dataset
    del confidence_scores_batch
    del confs
    
    gc.collect()
    
    return True
    


    

    
