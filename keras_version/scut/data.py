# -*- encoding:utf-8 -*-
import os
import cv2
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
    return (trainImageGen, valImageGen, testImageGen), (totalTrain, totalVal, totalTest)


def create_visit_gen(HEIGHT, WIDTH, CHANNEL):
    # derive the paths to the training, validation, and testing directories
    trainVisitPath = os.path.sep.join([config.BASE_PATH, config.BASE_VISIT_TYPE, config.TRAIN])
    validVisitPath = os.path.sep.join([config.BASE_PATH, config.BASE_VISIT_TYPE, config.VAL])
    testVisitPath = os.path.sep.join([config.BASE_PATH, config.BASE_VISIT_TYPE, config.TEST])

    return trainVisitGen, valVisitGen, testVisitGen


def load_data(filesPath, exts=('.jpg')):
    files = paths.list_files(filesPath, validExts=exts)
    datas = []
    label = []
    for fPath in files:
        l = fPath.split(os.path.sep)[-2]
        label.append(config.CLASSES.index(l))
        if 'jpg' in exts:
            im = cv2.imread(fPath)
            datas.append(im)
        elif 'npy' in exts:
            da = np.load(fPath)
            elm = np.pad(da, ((4,4), (3,3), (0,0)), mode='constant', constant_values=0) # 24x26x7 --> 32x32x7
            datas.append(elm)
        else:
            print('{}'.format(fPath))
    datas = np.array(datas)
    label = np.array(label)
    label = np.expand_dims(label, axis=1)
    return datas, label

def load_image_data():
    trainImagePath = os.path.sep.join([config.BASE_PATH, config.BASE_IMAGE_TYPE, config.TRAIN])
    validImagePath = os.path.sep.join([config.BASE_PATH, config.BASE_IMAGE_TYPE, config.VAL])
    testImagePath = os.path.sep.join([config.BASE_PATH, config.BASE_IMAGE_TYPE, config.TEST])
    trainImageData, trainImageLabel = load_data(trainImagePath)
    validImageData, validImageLabel = load_data(validImagePath)
    testImageData, testImageLabel = load_data(testImagePath)
    return (trainImageData, trainImageLabel), (validImageData, validImageLabel), (testImageData, testImageLabel)

def load_visit_data():
    # derive the paths to the training, validation, and testing directories
    trainVisitPath = os.path.sep.join([config.BASE_PATH, config.BASE_VISIT_TYPE, config.TRAIN])
    validVisitPath = os.path.sep.join([config.BASE_PATH, config.BASE_VISIT_TYPE, config.VAL])
    testVisitPath = os.path.sep.join([config.BASE_PATH, config.BASE_VISIT_TYPE, config.TEST])
    trainVisitData, trainVisitLabel = load_data(trainVisitPath, exts=('.npy'))
    validVisitData, validVisitLabel = load_data(validVisitPath, exts=('.npy'))
    testVisitData, testVisitLabel = load_data(testVisitPath, exts=('.npy'))
    return (trainVisitData, trainVisitLabel), (validVisitData, validVisitLabel), (testVisitData, testVisitLabel)


def create_data_gen(imagePath, visitPath, mode='train', bs=config.BATCH_SIZE, numClasses=len(config.CLASSES)):
    imageFiles = paths.list_files(imagePath, validExts=('.jpg'))
    visitFiles = paths.list_files(visitPath, validExts=('.npy'))
    imageIt = iter(imageFiles)
    visitIt = iter(visitFiles)
    while True:
        imageData = []
        visitData = []
        labels = []
        while(len(labels)<bs):
            try:
                iPath = next(imageIt)
                imageData.append(cv2.imread(iPath))
                l = iPath.split(os.path.sep)[-2]
                label = config.CLASSES.index(l)
                label = keras.utils.to_categorical(label, num_classes=numClasses)
                labels.append(label)

                vPath = next(visitIt)
                da = np.load(vPath) # 24x26x7
                elm = np.pad(da, ((4,4), (3,3), (0,0)), mode='constant', constant_values=0) # 32x32x7
                visitData.append(elm)
            except StopIteration:
                imageIt = iter(imageFiles)
                visitIt = iter(visitFiles)
                if mode in ('valid', 'test'):
                    break
        yield ([np.array(imageData), np.array(visitData)], np.array(labels))
