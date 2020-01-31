import numpy as np
import cv2

def decode_image(img):
    
    # Convert the bytes-data from the database into a numpy array
    np_img = np.frombuffer(img, dtype = np.uint8)
    
    # Decode the array back to an image
    image = cv2.imdecode(np_img, cv2.IMREAD_ANYCOLOR)
    
    return image