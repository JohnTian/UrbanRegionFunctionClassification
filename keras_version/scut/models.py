# -*- encoding:utf-8 -*-
import keras
import numpy as np
from scut import config
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense
from keras.applications import VGG16
from keras.layers.core import Flatten
from keras.layers.core import Dropout


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


def create_image_model(HEIGHT, WIDTH, CHANNEL):
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


def create_visit_model(HEIGHT, WIDTH, CHANNEL):
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