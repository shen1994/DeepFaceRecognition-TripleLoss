# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 10:11:26 2018

@author: shen1994
"""

from keras import backend as K
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, Lambda
from keras.legacy.layers import MaxoutDense

def lrn_layer(x, alpha=0.0005, beta=0.75, k=2):
    _, r, c, f = x.shape
    squared = K.square(x)
    pooled = K.pool2d(squared, (5, 5), strides=(1, 1), padding="same", pool_mode="avg")
    summed = K.sum(pooled, axis=3, keepdims=True)
    averaged = alpha * K.repeat_elements(summed, f, axis=3)
    denom = K.pow(k + averaged, beta)
    
    return x / denom
    
def lrn_output_shape(input_shape):
    
    return input_shape

def facenet_cnn(image_shape, embedding_size=128, dropout=0.8):
    
    model = Sequential()
    
    model.add(Convolution2D(filters=64, kernel_size=7, strides=2, input_shape=image_shape, padding='same', activation='relu', name='vector'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
    model.add(Lambda(lrn_layer, output_shape=lrn_output_shape))
    
    model.add(Convolution2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu'))
    model.add(Convolution2D(filters=192, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
    model.add(Lambda(lrn_layer,output_shape=lrn_output_shape))
    
    model.add(Convolution2D(filters=192, kernel_size=1, strides=1, padding='same', activation='relu'))
    model.add(Convolution2D(filters=384, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
    
    model.add(Convolution2D(filters=384, kernel_size=1, strides=1, padding='same', activation='relu'))
    model.add(Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
    
    model.add(Convolution2D(filters=256, kernel_size=1, strides=1, padding='same', activation='relu'))
    model.add(Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
    
    model.add(Convolution2D(filters=256, kernel_size=1, strides=1, padding='same', activation='relu'))
    model.add(Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
    
    model.add(Flatten())
    model.add(Dropout(dropout))
    
    model.add(MaxoutDense(32*embedding_size, 2))
    model.add(MaxoutDense(32*embedding_size,2))
    model.add(Dense(embedding_size))
    
    model.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))
    
    return model
    