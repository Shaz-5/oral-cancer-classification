import os
import shutil
import pathlib

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow import keras
import tensorflow
from tensorflow.keras import layers

from tensorflow.keras.utils import image_dataset_from_directory

import matplotlib.pyplot as plt

import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy.ndimage

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

train_datagen = ImageDataGenerator(
    rescale=1./255,   # normalization
    shear_range=15,   # shearing [-15°, 15°]
    width_shift_range=0.15,  # translation [-15, 15]
    height_shift_range=0.15,
    rotation_range=25,  # rotation [-25°, 25°]
    zoom_range=0.2,     # zoom augmentation
    horizontal_flip=True,  # flip augmentation
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
        'oral_cancer_train_test_split/train/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

validation_data = test_datagen.flow_from_directory(
        'oral_cancer_train_test_split/test/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
        

from tensorflow.keras.applications import DenseNet121

base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)  
x = Dropout(0.5)(x)  
x = Dense(1024, activation='relu')(x)  
predictions = Dense(3, activation='softmax')(x)

densenet_model = Model(inputs=base_model.input, outputs=predictions)

densenet_model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
              
callbacks = [ keras.callbacks.ModelCheckpoint( filepath="densenet.keras", save_best_only=True, monitor="val_loss") ]

history = densenet_model.fit(
    train_data,
    validation_data=validation_data,
    epochs=50,
    steps_per_epoch=train_data.samples // train_data.batch_size,
    validation_steps=validation_data.samples // validation_data.batch_size,
    callbacks=callbacks
)

test_loss, test_acc = densenet_model.evaluate(validation_data)
print(f"Test accuracy: {test_acc:.3f}")

np.save('densenet_history.npy',history)
# history = np.load('densenet_history.npy',allow_pickle='TRUE').item()

test_model = keras.models.load_model("densenet.keras")
test_loss, test_acc = test_model.evaluate(validation_data)  
print(f"Test accuracy: {test_acc:.3f}")
