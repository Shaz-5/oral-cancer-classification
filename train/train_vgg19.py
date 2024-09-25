import os
import shutil
import pathlib

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.utils import image_dataset_from_directory

import matplotlib.pyplot as plt

import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy.ndimage

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications import VGG19

train_datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

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
        

base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x) 
x = Dense(1024, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)

vgg19_model = Model(inputs=base_model.input, outputs=predictions)

vgg19_model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# vgg19_model.summary()

# train vgg19
callbacks = [ keras.callbacks.ModelCheckpoint( filepath="vgg19.keras", save_best_only=True, monitor="val_loss") ]

history = vgg19_model.fit(
    train_data,
    validation_data=validation_data,
    epochs=50,
    steps_per_epoch=train_data.samples // train_data.batch_size,
    validation_steps=validation_data.samples // validation_data.batch_size,
    callbacks=callbacks
)

np.save('vgg19_history.npy',history)

test_model = keras.models.load_model("vgg19.keras")
test_loss, test_acc = test_model.evaluate(validation_data)
print(f"Test accuracy: {test_acc:.3f}")
