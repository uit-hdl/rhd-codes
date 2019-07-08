from keras.models import load_model
from keras.preprocessing import image
from os import walk
import numpy as np
import os
import glob
import sqlite3
import cv2

con = sqlite3.connect('D:/Ruby_Prosjekter/Bilde/Yrkeskoder.db3')	# Path to your database
cur = con.cursor()
f = open('tresiffersqlite_1000.txt', 'w') 							# File to save prediction results in
# path to model
model_path = 'Tresiffer_optimum_weightsVGG16_256_3D_1000.hdf5'
# dimensions of images
img_width, img_height = 496, 108 								

# load the trained model
model = load_model(model_path)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

row = cur.execute("SELECT Rad, Bilde from Bilder")
for ObjId, item in row:
	nparr  = np.fromstring(item, np.uint8)
	img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
	img_np = cv2.resize(img_np, (img_height, img_width))
	img = image.img_to_array(img_np)
	img = np.expand_dims(img, axis=0)
	#print(img.shape)



	classes = model.predict_classes(img, batch_size=20)
	output = ObjId + '\t' + str(classes[0])
	linje = [ObjId,classes[0]]
	#print(classes[0])
	print(output)
	f.write(output + '\n')

f.close