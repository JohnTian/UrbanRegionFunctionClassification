# -*- encoding:utf-8 -*-
import os
import cv2
import keras
import copy
import numpy as np
from scut.util import rotate, randomCropAndNormal
from scut import config
from keras.models import load_model


def tta_core(model, iP, vP, aug=True):
    # load original data
    im = randomCropAndNormal(iP)
    vp = np.load(vP)
    # core
    if aug:
        imlist = [
            im,
            rotate(im, 2), 
            rotate(im, 3), 
            rotate(im, -2), 
            rotate(im, -3)
        ]
        vilist = [vp] * len(imlist)
    else:
        imlist = [im]
        vilist = [vp]
    # save preds
    preds = []
    for im, vi in zip(imlist, vilist):
        im = np.expand_dims(im, axis=0)
        vi = np.expand_dims(vi, axis=0)
        pred = model.predict([im, vi])
        preds.append(pred)
    preds = np.mean(preds, axis=0)
    preds = np.argmax(preds, axis=-1)
    return preds


def TTA(mode='submit'):
    modelPath = os.path.sep.join([config.MODEL_PATH, config.MODEL_NAME])
    model = load_model(modelPath)
    numClasses = len(config.CLASSES)

    if mode != 'submit':
        testImgPath = os.path.sep.join([config.BASE_PATH, config.BASE_IMAGE_TYPE, config.TEST, 'testImage.txt'])
        testVisPath = os.path.sep.join([config.BASE_PATH, config.BASE_VISIT_TYPE, config.TEST, 'testVisit.txt'])
        ivpass = 0
        with open(testImgPath, 'r') as fi:
            testImgPaths = fi.readlines()
        with open(testVisPath, 'r') as fv:
            testVisPaths = fv.readlines()
        sumv = len(testImgPaths)
        for iP, vP in zip(testImgPaths, testVisPaths):
            iP = iP.strip('\n')
            vP = vP.strip('\n')
            # tta label
            pred = tta_core(model, iP, vP, True) # 0.619753
            # actual label
            l = vP.split(os.path.sep)[-2]
            label = config.CLASSES.index(l)
            if pred == label:
                ivpass = ivpass + 1
            else:
                print('[INFO] label: {} --> pred: {}'.format(label, pred))
        print('[INFO] test total {}, predict ratio {:.6f}'.format(sumv, 1.0*ivpass/sumv))


if __name__ == "__main__":
    TTA(mode='test')
