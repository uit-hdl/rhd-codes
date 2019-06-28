from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import sqlite3
import cv2

    
# Function to return the prediction values for all 10 classes
def prediction_values_toString(prediction_values):
    
    result = ''
    
    for value in prediction_values:
        result += str(value)
        result += '\t'
        
    return result

# Function to evaluate the confidence values for image prediction
def evaluate_confidence_scores(confidence_values):
    
    # Find max value, and second highest value. 
    # If there is a large difference (> 15%) between the max and second highest values, return just the max
    # If MAX - Second_Highest < threshold then too close to tell -> return both values
        
    # List of return values. Either [Max index, Max value] or [Max index, Max value, Second Highest index, Second Highest value]
    # If the image cannot be confidently classified, return a 0 to indicate, followed by value and prediction
    return_list = []
    
    #Threshold value, can be tuned
    threshold = 0.15    
    
    # Find the confidence of the max value, and it's index
    max_value = max(confidence_values)
    max_index = np.where(confidence_values == np.amax(confidence_values))[0][0]
    
    # If the best possible value (max) is less than 20%, the image cannot be classified confidently
    if max_value > 0.20:
            
        # Create a copy of the values, and remove the previous max from the copy
        temp_list = confidence_values.copy()
        temp_list = np.delete(temp_list, np.where(temp_list == np.amax(confidence_values)))    
        
        # Find the confidence of the second highest value, and it's index in the ORIGINAL list of confidence values
        second_highest_value = max(temp_list)
        second_highest_index = np.where(confidence_values == second_highest_value)[0][0]
        
        
        if (max_value - second_highest_value > threshold):
            return_list.append([max_index, max_value])
        
        else:
            return_list.append([max_index, max_value, second_highest_index, second_highest_value])
            
        return return_list
    
    else:
        return_list.append([999, max_value, max_index])
        return return_list
    
def image_centering(image):
    
    num_rows, num_cols = image.shape[:2]
    
    # Calculations
    ret, thresh = cv2.threshold(image, 127, 255, 0)
    
    M = cv2.moments(thresh)
    
    # To avoid ZeroDivisionError - will result in some images that are not centered, but should be relatively small percentage
    if M['m10'] == 0:
        return image
    elif M['m01'] == 0:
        return image
    elif M['m00'] == 0:
        return image
    
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    
    # Centering
    distance_to_center_X = 0
    distance_to_center_Y = 0
    
    if cX < 50:
        distance_to_center_X = 50 - cX      # Needs to move right / X+
    elif cX > 50:
        distance_to_center_X = -(cX - 50)      # Needs to move left / X-
    else:
        distance_to_center_X = 0
        
    if cY < 50:
        distance_to_center_Y = 50 - cY 
    elif cX > 50:
        distance_to_center_Y = -(cY - 50)
    else:
        distance_to_center_Y = 0
    
    # Translations
    translation_matrix = np.float32([ [1, 0, distance_to_center_X], [0, 1, distance_to_center_Y] ])      # Lucky. Know we need to translate 35 in X+ direction
    img_translated = cv2.warpAffine(image, translation_matrix, (num_cols, num_rows))
    
    return img_translated

# Remove previous version of output file
if os.path.exists('ensiffersqlite2.txt'):
    os.remove('ensiffersqlite2.txt')


con = sqlite3.connect('Z:/UtklippsDatabaser/YrkeskoderES.db3')
cur = con.cursor()
# path to model
model_path = 'Centered_BalancedTraining_BalancedValidation_sparseCategoricalCrossentropy_ESValAcc_SimpleTrainGen_Model.HDF5'
# dimensions of images
img_width, img_height = 100, 100

# load the trained model
model = load_model(model_path)

test_datagen = ImageDataGenerator(rescale=1./255)

old_query = 'SELECT Rad, Bilde from Bilder limit 100'

row = cur.execute(old_query)
for ObjId, item in row:                                                         # ObjID is the row, item is the image
    #print(files[x])
    nparr  = np.fromstring(item, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img_np = cv2.resize(img_np, (img_height, img_width))

# =============================================================================
#     #Display image
#     cv2.imshow('imgage', img_np)
#     k = cv2.waitKey(0)
#     if k == 27:
#         cv2.destroyAllWindows()
# =============================================================================

    img = image.img_to_array(img_np)

    # center the image
    img = image_centering(img)
        
    #Reshape image for prediction
    img = img.reshape(1, img.shape[0], img.shape[1], 1)
    
    # Prediction
    for batch in test_datagen.flow(img, batch_size=1):
        pred = model.predict(batch)
        
        predictions = pred.reshape(pred.shape[1])
        
        filename = ObjId
        text = ''
                
        confidence_values = evaluate_confidence_scores(predictions)
        
        # Flag that indicates if the model was unsure when assigning the label to the prediction. Unsure = 15% or less difference
        flag = 0
        
        # Checking to see if the predicted values for at least 2 classes are very close. i.e the model was very unsure with this image
        if len(confidence_values) > 2:
            flag = 1
        
        # The values that will be printed: Filename     Label       prediction_values for all labels    flag that indicates if the model was unsure about the prediction
        text = filename + '\t' + str(confidence_values[0][0]) + '\t' + prediction_values_toString(predictions) + '\t' + str(flag)
        
        # Check if image was able to be classified confidently (With a confidence score above 20%)
        # If not, the output string is replaced with XXXX
        if confidence_values[0] == 999:
            text = 'XXXX\n'      # This image could not be classified with a confidence of over 20%
        
        # Write prediction to file                
        with open('ensiffersqlite2.txt', 'a') as f:
            f.write(text + '\n')
            
        # Break out of looping over the ImageDataGenerator batch - move to the next image
        break
       
