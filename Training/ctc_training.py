# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 12:49:20 2021

@author: bpe043
"""

import os
import numpy as np
import pandas as pd

from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Creating the CTC layer as our final layer, so we can use the ctc loss function
class CTCLayer(layers.Layer):
    def __init__(self, name = None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost
        
    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it to the layer using 'self.add_loss()'
        batch_len = tf.cast(tf.shape(y_true)[0], dtype = "int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype = "int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype = "int64")
        
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype = "int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype = "int64")
        
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        
        # At test time, just return the computed predictions
        return y_pred
    

# Creating the model
def build_model():
    # Inputs to the model
    input_img = layers.Input(
            shape= (img_width, img_height, 1), name = "image", dtype = "float32"
    )
    labels = layers.Input(name = "label", shape = (None, ), dtype = "float32")
    
    # First conv block
    x = layers.Conv2D(
            32,
            (3, 3),
            activation = "relu",
            kernel_initializer = "he_normal",
            padding = "same",
            name = "Conv1"
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name = "pool1")(x)
    
    # Second conv block
    x = layers.Conv2D(
            64,
            (3, 3),
            activation = "relu",
            kernel_initializer = "he_normal",
            padding = "same",
            name = "Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name = "pool2")(x)
    
    # We have used two max pool layers with pool size and strides 2.
    # Hence, downsampled feature maps are 4 times smaller. The number of filters in the last layer is 64.
    # Reshape accordingly before passing the output to the RNN part of the model
    new_shape = ((img_width // downsample_factor), (img_height // downsample_factor) * 64)
    x = layers.Reshape(target_shape = new_shape, name = "reshape")(x)
    x = layers.Dense(64, activation = "relu", name = "dense1")(x)
    x = layers.Dropout(0.2)(x)
    
    # RNN part
    x = layers.Bidirectional(layers.LSTM(128, return_sequences = True, dropout = 0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences = True, dropout = 0.25))(x)
    
    # Output layer
    x = layers.Dense(len(characters) + 1, activation = "softmax", name = "dense2")(x)
    
    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name = "ctc_loss")(labels, x)
    
    # Define the model
    model = keras.models.Model(
            inputs = [input_img, labels], outputs = output, name = "ocr_model_v1"
    )
    
    # Optimizer
    opt = keras.optimizers.Adam()
    
    # Compile the model and return
    model.compile(optimizer = opt)
    
    # Will let you set breakpoints in backend functions
    model.run_eagerly = True
    
    return model


    
    

# Splitting the data into training and validation
def split_data(images, labels, train_size = 0.9, shuffle=True):
    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    
    return x_train, x_valid, y_train, y_valid


# Getting the images into the proper form for prediction
def encode_single_sample(img_path, label):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 7. Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}


# A utility function to decode the output of the network
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


    
# Your images should be stored in a folder
data_dir = Path("<Path_to_your_images>")
images = sorted(list(map(str, list(data_dir.glob("*.jpg")))))
labels = [img.split(os.path.sep)[-1].split("-")[0] for img in images]
names = [x.split('-')[1].split('.')[0] for x in images]

characters = sorted(set(char for label in labels for char in label))

# Store characters. Because the order they appear in here will need to be the same they appear in when doing inference
with open('characters.txt', 'w') as f:
    f.write(', '.join(characters))

print("Number of images found: ", len(images))
print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)

# Desired image dimensions
img_width = 200
img_height = 115


# Factor by which the image is going to be downsampled
# by the convolutional blocks. We will be using two
# convolution blocks and each block will have
# a pooling layer which downsample the features by a factor of 2.
# Hence total downsampling factor would be 4.
# If we go with a bigger model, change the "two" to reflect the number of pooling layers we end up using
# Two * (2,2) => 2 * 2 = 4
downsample_factor = 4

# Maximum length of any label in the dataset
max_length = max([len(label) for label in labels])

# Mapping characters to integers
char_to_num = layers.experimental.preprocessing.StringLookup(vocabulary = list(characters), num_oov_indices=0, mask_token = None)

# Mapping integers back to original characters
num_to_char = layers.experimental.preprocessing.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)


# Splitting data into training and validation sets
x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))


# Create the training and validation tensors

# Batch size for training and validation
batch_size = 16

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    train_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation_dataset = (
    validation_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)
    
        
        
# Get the model
model = build_model()
model.summary()

# Training
epochs = 100
early_stopping_patience = 10

# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
        monitor = "val_loss", patience = early_stopping_patience, restore_best_weights = True
)

# Train the model
history = model.fit(
        train_dataset,
        validation_data = validation_dataset,
        epochs = epochs,
        callbacks=[early_stopping],
)

# Inference

# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
        model.get_layer(name = "image").input, model.get_layer(name = "dense2").output
)
prediction_model.summary()

prediction_model.save("<Path_to_store_the_model>")
      

# Lists used to determine number of "correct" and "uncorrect" predictions
# Stores batch number, image number, actual_y, predicted_y
matches = []
misses = []

# Various indexes used for the lists
valid_indexing = 0
pred_indexing = 0
batch_index = 0

# Create a dataframe to hold confidence scores for all validaiton images
confidence_scores = pd.DataFrame(columns = ['Image_name', 'Predicted_Label', 'Actual_Label', 'C0', 'C1', 'C2'])

# Create a folder to store the results from the prediction
if not os.path.exists('Results_folder'):
    os.makedirs('Results_folder')

# Check results on some validation samples
# take(-1) gives All the images in the validation/training/whatever set, but in batches of the batch-size we've given
# For the example, the validation set, the batch size is 16, so validation_dataset.take(1) gives One batch of 16 images/labels.
# If the validation set consists of 104 images, then validation_dataset.take(-1) gives Six batches of 16, and One batch of 8. (16*6 + 8 = 104)
for batch in validation_dataset.take(-1):
    batch_images = batch['image']
    batch_labels = batch['label']
    
    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)
    
    # Create a dataframe to hold the confidence scores of the current image batch
    confidence_scores_batch = pd.DataFrame(index = np.arange(len(pred_texts)), columns = ['Image_name', 'Predicted_Label', 'Actual_Label', 'C0', 'C1', 'C2'])
    
    # Check pred_texts up against the same indexes from y_valid
    pred_indexing += len(pred_texts)
    batch_validation = list(y_valid[valid_indexing : pred_indexing])
    
    # With the same way we grab the actual labels, we also grab the image names for storing
    image_names = list(x_valid[valid_indexing : pred_indexing])
    image_names = [x.split('-')[1] for x in image_names]
    
    valid_indexing += len(pred_texts)
    
    v_index = 0
    for v in batch_validation:
        v = str(v)
        if v != pred_texts[v_index]:
            missmatch = (batch_index, v_index, v, pred_texts[v_index])
            misses.append(missmatch)
        else:
            match = (batch_index, v_index, v, pred_texts[v_index])
            matches.append(match)
            
        v_index += 1
    
    # Convert back to the original text strings from the mapped values used in the CTC loss function
    orig_texts = []
    for label in batch_labels:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        orig_texts.append(label)
        
    
    # Process predictions
    for i in range(len(pred_texts)):
        
        # Store confidence scores
        conf_start = preds[i].max(axis = 1)
        lab_start = np.argmax(preds[i], axis = 1)
        confs = np.delete(conf_start, np.where(lab_start == 13))
        
        # If we get less than 3 confidence scores, we don't have enough information to say that this is a valid prediction
        if len(confs) == 3:
            img_confs = (confs[0], confs[1], confs[2])
            
            # Now that we have the label from pred_texts and the confidence scores for the labels in img_confs, we can add them to our batch dataframe
            confidence_scores_batch.loc[i] = [image_names[i], pred_texts[i], batch_validation[i], float(img_confs[0]), float(img_confs[1]), float(img_confs[2])]
            
        
        
        # Store image in a folder with the prediction being the 
        img = (batch_images[i] * 255).numpy().astype("uint8")
        img = img[:, :, 0].T
        lbl = pred_texts[i] + '.png'
        lbl = "Results_folder//" + str(batch_index) + "//image_nr_" + str(i) + '_' + lbl
        cv2.imwrite(lbl, img)
        

    # At the end of the batch, we store the batch dataframe, and we also append it to our total dataframe
    confidence_scores_batch.to_csv("Results_folder//{}//confidence_scores.csv".format(batch_index), sep = ';', encoding = "utf-8")
    confidence_scores = confidence_scores.append(confidence_scores_batch, ignore_index = True)
    
    print(batch_index)
    batch_index += 1
    

# After validation, we store our total dataframe
# We also add them image names of the validation images, x_val correponds 1-to-1 with the order of the predictions, so we just add
confidence_scores.to_csv("Results_folder//total_confidence_scores.csv", sep = ';', encoding = "utf-8")

    
        
        





