# -*- encoding:utf-8 -*-
import os
import time
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import to_categorical

def validTest(validFile, modelPath, num_classes):
    start_time = time.time()
    validVisit = np.load('data/npy/validVisit.npy')
    validImage = np.load('data/npy/validImage.npy')
    validLabel = np.load('data/npy/validLabel.npy')
    allnum = validVisit.shape[0]
    print("[INFO] load data using time:%.2fs"%(time.time()-start_time))
    print("[INFO] number of load  data:%d"%allnum)

    validLabel = np.reshape(validLabel, (validLabel.shape[0], 1)) - 1       # 0 - 8
    validLabel = to_categorical(validLabel, num_classes=num_classes)        # one-hot
    validClass = np.argmax(validLabel, axis=1)                              # argmax 0-8

    # 加载模型
    model = load_model(modelPath)

    num = 0
    if os.path.isfile(validFile) and os.path.exists(validFile):
        os.remove(validFile)
        print('[INFO] remove if exits.')
    start_time = time.time()
    with open(validFile, 'w+') as fi:
        for i in range(allnum):
            label = validClass[i]
            preLabel = model.predict([validVisit[i:i+1], validImage[i:i+1]])
            preClass = np.argmax(preLabel, axis=1)[0]
            if label == preClass:
                num += 1
            fi.write(str(label+1).zfill(6) + ' \t ' + str(preClass+1).zfill(6) + '\n')
    print("[INFO] predict using time:%.2fs"%(time.time()-start_time))
    print('[INFO] reg num : %d' % num)
    print('[INFO] test num: %d' % allnum)
    print('[INFO] reg rate: %f' % (1.0*num/allnum))


def evalTest(evalFile, modelPath):
    start_time = time.time()
    evalVisit = np.load('data/npy/evalVisit.npy')
    evalImage = np.load('data/npy/evalImage.npy')
    allnum = evalVisit.shape[0]
    print("[INFO] load data using time:%.2fs"%(time.time()-start_time))
    print("[INFO] number of load  data:%d"%evalVisit.shape[0])

    model = load_model(modelPath)

    if os.path.exists(evalFile):
        os.remove(evalFile)
    start_time = time.time()
    with open(evalFile, 'w+') as fi:
        for i in range(allnum):
            preLabel = model.predict([evalVisit[i:i+1], evalImage[i:i+1]])
            preClass = np.argmax(preLabel, axis=1)[0]
            fi.write("%s \t %03d\n"%(str(i).zfill(6), preClass+1))
    print("[INFO] predict using time:%.2fs"%(time.time()-start_time))
    print('[INFO] test num: %d' % allnum)


if __name__ == "__main__":
    num_classes = 9
    modelPath = 'saved_models/urban_ResNet20v1_model.084.h5'
    validTest('urban_ResNet20v1_model.084.h5.valid.txt', modelPath, num_classes)
    evalTest('urban_ResNet20v1_model.084.h5.txt', modelPath)
