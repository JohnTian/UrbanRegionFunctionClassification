# -*- encoding:utf-8 -*-
import keras
import numpy as np
from scut import config
from .resnet import create_resnet
from keras.models import Model
from keras.layers import MaxPooling2D, SeparableConvolution2D
from keras.layers import Input, Conv2D, Activation, BatchNormalization
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


def create_image_model(HEIGHT, WIDTH, CHANNEL):
    # load the base network, ensuring the head FC layer sets are left off
    baseModel = DenseNet201(weights="imagenet", include_top=False, input_tensor=Input(shape=(HEIGHT, WIDTH, CHANNEL)))

    # construct the head of the model that will be placed on top of the the base model
    headModel = baseModel.output
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(49, activation="relu")(headModel)
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
    ## Version 1.1
    # return create_resnet((HEIGHT, WIDTH, CHANNEL))

    ## Version 1.2
    filters=(16, 32, 64)
    inputShape = (HEIGHT, WIDTH, CHANNEL)
    chanDim = -1
	# define the model input
    inputs = Input(shape=inputShape)
	# loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs

        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)

    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model
