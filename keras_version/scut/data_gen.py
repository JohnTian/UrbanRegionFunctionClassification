# -*- encoding:utf-8 -*-
import os
import keras
import pickle
import numpy as np
from scut import config
from imutils import paths
from keras.preprocessing.image import ImageDataGenerator


def create_image_gen(HEIGHT, WIDTH, CHANNEL):
    # derive the paths to the training, validation, and testing directories
    trainImagePath = os.path.sep.join([config.BASE_PATH, config.BASE_IMAGE_TYPE, config.TRAIN])
    validImagePath = os.path.sep.join([config.BASE_PATH, config.BASE_IMAGE_TYPE, config.VAL])
    testImagePath = os.path.sep.join([config.BASE_PATH, config.BASE_IMAGE_TYPE, config.TEST])

    trainVisitPath = os.path.sep.join([config.BASE_PATH, config.BASE_VISIT_TYPE, config.TRAIN])
    validVisitPath = os.path.sep.join([config.BASE_PATH, config.BASE_VISIT_TYPE, config.VAL])
    testVisitPath = os.path.sep.join([config.BASE_PATH, config.BASE_VISIT_TYPE, config.TEST])

    # determine the total number of image paths in training, validation and testing directories
    totalTrain = len(list(paths.list_images(trainImagePath)))
    totalVal = len(list(paths.list_images(valImagePath)))
    totalTest = len(list(paths.list_images(testImagePath)))

    # initialize the training data augmentation object
    trainImageAug = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    # initialize the validation/testing data augmentation object (which we'll be adding mean subtraction to)
    valImageAug = ImageDataGenerator()
    # define the ImageNet mean subtraction (in RGB order) and set the
    # the mean subtraction value for each of the data augmentation
    # objects
    mean = np.array([123.68, 116.779, 103.939], dtype="float32")
    trainImageAug.mean = mean
    valImageAug.mean = mean

    TARGET_SIZE = (HEIGHT, WIDTH)
    # initialize the training generator
    trainImageGen = trainImageAug.flow_from_directory(
        trainImagePath,
        class_mode="categorical",
        target_size=TARGET_SIZE,
        color_mode="rgb",
        shuffle=True,
        batch_size=config.BATCH_SIZE)

    # initialize the validation generator
    valImageGen = valImageAug.flow_from_directory(
        valImagePath,
        class_mode="categorical",
        target_size=TARGET_SIZE,
        color_mode="rgb",
        shuffle=False,
        batch_size=config.BATCH_SIZE)

    # initialize the testing generator
    testImageGen = valImageAug.flow_from_directory(
        testImagePath,
        class_mode="categorical",
        target_size=TARGET_SIZE,
        color_mode="rgb",
        shuffle=False,
        batch_size=config.BATCH_SIZE)
    return trainImageAug, valImageGen, testImageGen

def create_visit_gen(HEIGHT, WIDTH, CHANNEL):
    return trainVistAug, valVisitGen, testVisitGen