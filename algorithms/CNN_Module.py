#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 19:13:07 2019

@author: HIT data mining lab
"""

# transfer G2 turbines(old) to G1 turbien(new) & finetuned experiments
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D, Dense, Flatten, Reshape, GlobalAveragePooling2D
from keras.layers import Dropout, Activation, BatchNormalization, ActivityRegularization
from keras import optimizers
from keras.optimizers import Adam,SGD
from keras.callbacks import Globalbest
from keras import regularizers


from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing as skpp#加入sklearn归一化操作
from sklearn.model_selection import train_test_split

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt


# %% 模型搭建
def cnn_model(n_width=1,n_height=7,n_depth=1):

	finemodel = Sequential()
	finemodel.add(Reshape(target_shape=(n_width,n_height,n_depth), 
					  input_shape=(n_inputs,)))

	finemodel.add(Conv2D(filters=n_filters[0], kernel_size=[1,3], 
					 kernel_regularizer=regularizers.l2(0.0001),
					 padding='SAME'))
	finemodel.add(BatchNormalization())
	finemodel.add(Activation('relu'))

	finemodel.add(Conv2D(filters=n_filters[0], kernel_size=[1,3], 
					 kernel_regularizer=regularizers.l2(0.0001),
					 padding='SAME'))
	finemodel.add(BatchNormalization())
	finemodel.add(Activation('relu'))

	finemodel.add(MaxPooling2D(pool_size=(1,2), strides=(1,2)))
	#finemodel.add(BatchNormalization())
	finemodel.add(Conv2D(filters=n_filters[1], kernel_size=[1,3], 
					 kernel_regularizer=regularizers.l2(0.0001),
					 padding='SAME'))
	finemodel.add(BatchNormalization())
	finemodel.add(Activation('relu'))

	finemodel.add(Conv2D(filters=n_filters[1], kernel_size=[1,3], 
					 kernel_regularizer=regularizers.l2(0.0001),
					 padding='SAME'))
	finemodel.add(BatchNormalization())
	finemodel.add(Activation('relu'))

	#finemodel.add(GlobalAveragePooling2D())
	finemodel.add(Flatten())
	finemodel.add(Dense(units=10, activation='relu'))
	#,,name="new_dense1"
	#finemodel.add(Dense(units=10, activation='relu'))
	finemodel.add(Dense(units=n_classes, activation='softmax'))
	#,name="new_dense2"
	finemodel.summary()
	return finemodel

