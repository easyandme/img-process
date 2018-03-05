# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 00:03:49 2018

@author: kobayashi
"""
import os
import numpy as np
import glob
from PIL import Image
from keras.models import Model
from keras.layers import Activation, BatchNormalization, Dropout
from keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
import matplotlib.pyplot as plt

# import tensorflow as tf

# Define input image
base_path = os.path.dirname(__file__)
filelist = glob.glob(os.path.join(base_path, '*.tif'))

# Load image
img = Image.open(filelist[0])  # image can be selected from the list
width, height = img.size
if width != 190 or height != 190:
    img.thumbnail((190, 190), Image.ANTIALIAS)
img = np.asarray(img, dtype=np.float32).reshape(-1, 190, 190, 1)
img -= np.mean(img)


def AEmodel():
    input_img = Input(shape=(190, 190, 1), name='input_layer')
    x = Conv2D(128, (3, 3), padding='same', name='block1_conv2')(input_img)
    x = BatchNormalization(name='block1_BN')(x)
    x = Activation('relu', name='block1_act')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
    x = BatchNormalization(name='block2_BN')(x)
    x = Activation('relu', name='block2_act')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='block2_pool')(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), padding='same', name='block3_conv2')(x)
    x = BatchNormalization(name='block3_BN')(x)
    x = Activation('relu', name='block3_act')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='block3_pool')(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), padding='same', name='block4_conv2')(x)
    x = BatchNormalization(name='block4_BN')(x)
    x = Activation('relu', name='block4_act')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='block4_pool')(x)
    x = Dropout(0.25)(x)

    cx = GlobalAveragePooling2D(name='globalAve')(x)
    cx = Dropout(0.5)(cx)
    class_output = Dense(2, activation='softmax', name='class_output')(cx)

    x = UpSampling2D((2, 2), name='block7_upsample')(x)
    x = Conv2D(32, (3, 3), padding='same', name='block7_conv2')(x)
    x = BatchNormalization(name='block7_BN')(x)
    x = Activation('relu', name='block7_act')(x)
    x = Dropout(0.25)(x)

    x = UpSampling2D((2, 2), name='block8_upsample')(x)
    x = Conv2D(32, (3, 3), padding='same', name='block8_conv2')(x)
    x = BatchNormalization(name='block8_BN')(x)
    x = Activation('relu', name='block8_act')(x)
    x = Dropout(0.25)(x)

    x = UpSampling2D((2, 2), name='block9_upsample')(x)
    x = Conv2D(16, (3, 3), padding='same', name='block9_conv2')(x)
    x = BatchNormalization(name='block9_BN')(x)
    x = Activation('relu', name='block9_act')(x)
    x = Dropout(0.25)(x)

    x = UpSampling2D((2, 2), name='block10_upsample')(x)
    x = Conv2D(16, (3, 3), padding='same', name='block10_conv2')(x)
    x = BatchNormalization(name='block10_BN')(x)
    x = Activation('relu', name='block10_act')(x)
    x = Dropout(0.25)(x)
    decoded = Conv2D(1, (3, 3), name='decoder_output')(x)
    autoencoder = Model(inputs=input_img, outputs=[class_output, decoded])
    return autoencoder


# Create a model
autoencoder = AEmodel()
# Load weights
autoencoder.load_weights('ACbin_33x128fl128GA_weights.h5')
batch_size = 20
# Extract output
intermediate_layer_model = Model(inputs=autoencoder.input,
                                 outputs=autoencoder.get_layer('globalAve').output)
intermediate_layer_model.compile('sgd','mse')
# Output the latent layer
intermediate_output = intermediate_layer_model.predict(
    img, batch_size=batch_size, verbose=1)

# Plot features
plt.figure()
plt.xlabel('Feature')
plt.ylabel('Intensity (a.u.)')
plt.bar(np.arange(len(intermediate_output[0])), intermediate_output[0, :])
plt.show()

