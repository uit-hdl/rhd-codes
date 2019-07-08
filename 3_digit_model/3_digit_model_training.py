import os
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import  *
from keras.optimizers import  *
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.regularizers import l2

#To store model details in csv file
from keras.callbacks import CSVLogger

# Stop training once a maximum result have been achieved
# and avoid overfitting
from keras.callbacks import EarlyStopping

# Checkpoints
# The EarlyStopping is operating with a set "patience" value, meaning that the model
# will continue to run, even when getting Worse results, for X additional epochs
# Using Checkpoints, we will store the model that performed the best
from keras.callbacks import ModelCheckpoint

csv_logger = CSVLogger('log3.csv', append=False, separator=';')

es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=10)

# Saving the model
mc_path = "Tresiffer_optimum_weightsVGG16_256_3D_1000.hdf5"
mc = ModelCheckpoint(mc_path, monitor='val_acc', mode='max', save_best_only = True, verbose = 1)

batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2)
training_set = train_datagen.flow_from_directory('D:/Bilder/montage/3s/1000',target_size = (496, 108),batch_size = 32,class_mode = 'categorical', color_mode = 'grayscale', subset='training')
test_set = train_datagen.flow_from_directory('D:/Bilder/montage/3s/1000',target_size = (496, 108),batch_size = 32,class_mode = 'categorical', color_mode = 'grayscale', subset='validation')

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(496,108,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(25, activation='softmax'))		# Using the top 25 classes of 3 digit codes

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

STEP_SIZE_TRAIN=training_set.n//training_set.batch_size
STEP_SIZE_VALID=test_set.n//test_set.batch_size

# Number of epochs is arbitrary since EarlyStopping is being used.
# It should never reach epoch 50
model.fit_generator(generator=training_set, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=test_set, validation_steps=STEP_SIZE_VALID, epochs=50, callbacks=[csv_logger, es, mc])
model.summary()
