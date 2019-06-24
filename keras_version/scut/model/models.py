# -*- encoding:utf-8 -*-
import keras
import numpy as np
from .. import config
from .resnet import create_resnet
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input, Conv2D, BatchNormalization, Activation
from keras.layers import MaxPooling2D, SeparableConvolution2D
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.applications import ResNet50, DenseNet201, VGG16, VGG19


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def create_image_model_baseon_pretrained(HEIGHT, WIDTH, CHANNEL):
    # load the base network, ensuring the head FC layer sets are left off
    baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(HEIGHT, WIDTH, CHANNEL)))

    # construct the head of the model that will be placed on top of the the base model
    headModel = baseModel.output
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(512, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(len(config.CLASSES), activation="softmax")(headModel)

    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)

    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False
    return model


def create_image_model(HEIGHT, WIDTH, CHANNEL):
    # define the model input
    inputs = Input(shape=(HEIGHT, WIDTH, CHANNEL))
    # loop over the number of filters
    filters = (8,16,32)
    idxOfFilters = list(range(len(filters)))
    flagOfPool2D = (True,True,True)
    chanDim = -1
    for (i, f, t) in zip(idxOfFilters, filters, flagOfPool2D):
        # if this is the first CONV layer then set the input appropriately
        if i == 0:
            x = inputs
        #x = Conv2D(filters=f, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01))(x)
        x = SeparableConvolution2D(filters=f, kernel_size=3, padding='same')(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation('relu')(x)
        #x = Dropout(0.5)(x)
        if t:
            x = MaxPooling2D(pool_size=(2, 2))(x)
    #!!! Fixed structure for output !!!
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=16, activation='relu')(x)
    x = Dense(units=len(config.CLASSES), activation='softmax')(x)
    # construct the CNN
    return Model(inputs, x)


def create_visit_model(HEIGHT, WIDTH, CHANNEL):
    # define the model input
    inputs = Input(shape=(HEIGHT, WIDTH, CHANNEL))
    # loop over the number of filters
    filters = (8,16,32)
    idxOfFilters = list(range(len(filters)))
    flagOfPool2D = (True,False,False)
    chanDim = -1
    for (i, f, t) in zip(idxOfFilters, filters, flagOfPool2D):
        # if this is the first CONV layer then set the input appropriately
        if i == 0:
            x = inputs
        #x = Conv2D(filters=f, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(x)
        #x = SeparableConvolution2D(filters=f, kernel_size=3, padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = SeparableConvolution2D(filters=f, kernel_size=3, padding='same')(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation('relu')(x)
        #x = Dropout(0.5)(x)
        if t:
            x = MaxPooling2D(pool_size=(2, 2))(x)
    #!!! Fixed structure for output !!!
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=16, activation='relu')(x)
    x = Dense(units=len(config.CLASSES), activation='softmax')(x)
    # construct the model
    return Model(inputs, x)
