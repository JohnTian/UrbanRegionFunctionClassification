# -*- encoding:utf-8 -*-
import os
import cv2
import keras
import pickle
import numpy as np
from .util import save2txt
from collections import OrderedDict
from .config import BASE_PATH, BASE_IMAGE_TYPE, BASE_VISIT_TYPE
from .config import TRAIN, VAL, TEST, BATCH_SIZE, CLASSES
from imutils import paths
from keras.preprocessing.image import ImageDataGenerator


def create_image_gen(HEIGHT, WIDTH, CHANNEL):
    # derive the paths to the training, validation, and testing directories
    trainImagePath = os.path.sep.join([BASE_PATH, BASE_IMAGE_TYPE, TRAIN])
    validImagePath = os.path.sep.join([BASE_PATH, BASE_IMAGE_TYPE, VAL])
    testImagePath = os.path.sep.join([BASE_PATH, BASE_IMAGE_TYPE, TEST])

    # determine the total number of image paths in training, validation and testing directories
    totalTrain = len(list(paths.list_images(trainImagePath)))
    totalVal = len(list(paths.list_images(validImagePath)))
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
        batch_size=BATCH_SIZE)

    # initialize the validation generator
    valImageGen = valImageAug.flow_from_directory(
        validImagePath,
        class_mode="categorical",
        target_size=TARGET_SIZE,
        color_mode="rgb",
        shuffle=False,
        batch_size=BATCH_SIZE)

    # initialize the testing generator
    testImageGen = valImageAug.flow_from_directory(
        testImagePath,
        class_mode="categorical",
        target_size=TARGET_SIZE,
        color_mode="rgb",
        shuffle=False,
        batch_size=BATCH_SIZE)
    return (trainImageGen, valImageGen, testImageGen), (totalTrain, totalVal, totalTest)


def load_data(filesPath, exts=('.jpg')):
    files = paths.list_files(filesPath, validExts=exts)
    datas = []
    label = []
    for fPath in files:
        l = fPath.split(os.path.sep)[-2]
        label.append(CLASSES.index(l))
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
    trainImagePath = os.path.sep.join([BASE_PATH, BASE_IMAGE_TYPE, TRAIN])
    validImagePath = os.path.sep.join([BASE_PATH, BASE_IMAGE_TYPE, VAL])
    testImagePath = os.path.sep.join([BASE_PATH, BASE_IMAGE_TYPE, TEST])
    trainImageData, trainImageLabel = load_data(trainImagePath)
    validImageData, validImageLabel = load_data(validImagePath)
    testImageData, testImageLabel = load_data(testImagePath)
    return (trainImageData, trainImageLabel), (validImageData, validImageLabel), (testImageData, testImageLabel)

def load_visit_data():
    # derive the paths to the training, validation, and testing directories
    trainVisitPath = os.path.sep.join([BASE_PATH, BASE_VISIT_TYPE, TRAIN])
    validVisitPath = os.path.sep.join([BASE_PATH, BASE_VISIT_TYPE, VAL])
    testVisitPath = os.path.sep.join([BASE_PATH, BASE_VISIT_TYPE, TEST])
    trainVisitData, trainVisitLabel = load_data(trainVisitPath, exts=('.npy'))
    validVisitData, validVisitLabel = load_data(validVisitPath, exts=('.npy'))
    testVisitData, testVisitLabel = load_data(testVisitPath, exts=('.npy'))
    return (trainVisitData, trainVisitLabel), (validVisitData, validVisitLabel), (testVisitData, testVisitLabel)


def preprocessing_data_gen():
    data = OrderedDict()
    trainImagePath = os.path.sep.join([BASE_PATH, BASE_IMAGE_TYPE, TRAIN])
    trainVisitPath = os.path.sep.join([BASE_PATH, BASE_VISIT_TYPE, TRAIN])
	# ------------------------------------  训练集均衡化 -- 均衡各个类的图像个数  ------------------------------------
    trainData = {}
	for code in CLASSES:
		trainData[code] = []
    for filePath in paths.list_images(trainImagePath):
        code = filePath.split(os.path.sep)[-2]
        trainData[code].append(filePath)
	trainDict = {}
	for code, value in trainData.items():
		trainDict[code] = len(value)
	maxNum = max(trainDict.values())
	for label in trainDict.keys():
		for _ in range(maxNum - len(trainData[label])):
			# 随机从均衡化前的0-trainDict[label] - 1区间内采样添加
			trainData[label].append(trainData[label][random.randint(0, trainDict[label] - 1)])
	if True:
		print('[DEBUG] after balance trainData...')
		for k, v in trainData.items():
			print('[DEBUG] after balance TrainData {:>25}:{:>5}'.format(k, len(v)))
		print('[DEBUG] after balance TrainData sum:', sum([len(v) for v in trainData.values()]))
    trainImageTxtPath = os.path.sep.join([trainImagePath, 'trainImage.txt'])
    trainVisitTxtPath = os.path.sep.join([trainVisitPath, 'trainVisit.txt'])
    totalTrain = 0
    fi = open(trainImageTxtPath, 'w+')
    fv = open(trainVisitTxtPath, 'w+')
    for k, v in trainData.items():
        for fP in v:
            totalTrain += 1
            fi.write(fP+'\n')
            fvP = fP.replace('image', 'visit').replace('.jpg', '.npy')
            fv.write(fvP+'\n')
    fi.close()
    fv.close()
    print('[INFO] totalTrain:', totalTrain)
    # ------------------------------------------------------------------------------------------------------------

    validImagePath = os.path.sep.join([BASE_PATH, BASE_IMAGE_TYPE, VAL])
    testImagePath = os.path.sep.join([BASE_PATH, BASE_IMAGE_TYPE, TEST])

    validVisitPath = os.path.sep.join([BASE_PATH, BASE_VISIT_TYPE, VAL])
    testVisitPath = os.path.sep.join([BASE_PATH, BASE_VISIT_TYPE, TEST])

    totalVal = len(list(paths.list_images(validImagePath)))

    testFiles = list(paths.list_images(testImagePath))
    testLabels = [CLASSES.index(line.split(os.path.sep)[-2]) for line in testFiles]
    testNames = [line.split(os.path.sep)[-2] for line in testFiles]
    totalTest = len(testLabels)

    # Core
    validImageTxtPath = os.path.sep.join([validImagePath, 'validImage.txt'])
    testImageTxtPath = os.path.sep.join([testImagePath, 'testImage.txt'])
    save2txt(validImagePath, ('.jpg'), validImageTxtPath)
    save2txt(testImagePath, ('.jpg'), testImageTxtPath)

    validVisitTxtPath = os.path.sep.join([validVisitPath, 'validVisit.txt'])
    testVisitTxtPath = os.path.sep.join([testVisitPath, 'testVisit.txt'])
    save2txt(validVisitPath, ('.npy'), validVisitTxtPath)
    save2txt(testVisitPath, ('.npy'), testVisitTxtPath)

    # Construct dict
    data['totalTrain'] = totalTrain
    data['totalVal'] = totalVal
    data['totalTest'] = totalTest
    data['testLabels'] = testLabels
    data['testNames'] = testNames
    data['trainImagePath'] = trainImageTxtPath
    data['validImagePath'] = validImageTxtPath
    data['testImagePath'] = testImageTxtPath
    data['trainVisitPath'] = trainVisitTxtPath
    data['validVisitPath'] = validVisitTxtPath
    data['testVisitPath'] = testVisitTxtPath

    return data


def create_data_gen(imagePath, visitPath, mode='train', bs=BATCH_SIZE, numClasses=len(CLASSES)):
    fImage = open(imagePath, 'r')
    fVisit = open(visitPath, 'r')
    while True:
        imageData = []
        visitData = []
        labels = []
        while len(imageData)<bs:
            iPath = fImage.readline().strip('\n')
            vPath = fVisit.readline().strip('\n')
            if (iPath == "") and (vPath == ""):
                fImage.seek(0)
                iPath = fImage.readline().strip('\n')
                fVisit.seek(0)
                vPath = fVisit.readline().strip('\n')
                if mode != 'train':
                    break
            # append image data
            imageData.append(cv2.imread(iPath))
            # append visit data
            # 24x26x7 --> 32x32x7    
            da = np.load(vPath)
            # elm = np.pad(da, ((4,4), (3,3), (0,0)), mode='constant', constant_values=0)
            visitData.append(da)
            # append label data
            l = vPath.split(os.path.sep)[-2]
            label = CLASSES.index(l)
            label = keras.utils.to_categorical(label, num_classes=numClasses)
            labels.append(label)
        yield ([np.array(imageData), np.array(visitData)], np.array(labels))


def image_gen(imagePath, mode='train', bs=BATCH_SIZE, numClasses=len(CLASSES)):
    fImage = open(imagePath, 'r')
    while True:
        imageData = []
        labels = []
        while len(imageData)<bs:
            iPath = fImage.readline().strip('\n')
            if iPath == "":
                fImage.seek(0)
                iPath = fImage.readline().strip('\n')
                if mode != 'train':
                    break
            # append image data
            imageData.append(cv2.imread(iPath))
            # append label data
            l = iPath.split(os.path.sep)[-2]
            label = CLASSES.index(l)
            label = keras.utils.to_categorical(label, num_classes=numClasses)
            labels.append(label)
        yield (np.array(imageData), np.array(labels))


def visit_gen(visitPath, mode='train', bs=BATCH_SIZE, numClasses=len(CLASSES)):
    fVisit = open(visitPath, 'r')
    while True:
        visitData = []
        labels = []
        while len(visitData)<bs:
            vPath = fVisit.readline().strip('\n')
            if vPath == "":
                fVisit.seek(0)
                vPath = fVisit.readline().strip('\n')
                if mode != 'train':
                    break
            # append visit data
            da = np.load(vPath)
            visitData.append(da)
            # append label data
            l = vPath.split(os.path.sep)[-2]
            label = CLASSES.index(l)
            label = keras.utils.to_categorical(label, num_classes=numClasses)
            labels.append(label)
        yield (np.array(visitData), np.array(labels))