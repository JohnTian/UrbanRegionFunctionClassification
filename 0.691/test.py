#encoding:utf-8
from __future__ import print_function
import os 
import time 
import json 
import torch 
import random 
import warnings
import torchvision
import numpy as np

from utils import *
from multimodal import MultiModalDataset,MultiModalNet,CosineAnnealingLR

from config import config
from datetime import datetime
from torch import nn,optim
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as F
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

from imutils import paths
from imgaug import augmenters as iaa
from torchvision import transforms as T

def ttaug(image):
    augment_img = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.GaussianBlur((0, 1.0)),
        iaa.AdditiveGaussianNoise(
                loc=0, scale=(0.0, 0.05*255), per_channel=0.5
            ),
        # iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
        #iaa.Multiply((1.2, 1.5)),
        iaa.SomeOf((0,4),[
            iaa.Affine(rotate=90),
            iaa.Affine(rotate=180),
            iaa.Affine(rotate=270),
            iaa.Affine(shear=(-16, 16)),
        ]),
        iaa.OneOf([
                #iaa.Crop(px=(0, 12)),
                iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                #iaa.Fliplr(0.5),
                iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
            ]),
        #iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
        #             children=iaa.WithChannels(0, iaa.Add(10))),
        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
        ], random_order=True)
    image_aug = augment_img.augment_image(image)
    return image_aug

def ttacore(model, iPath, vPath, b_debug=False):
    im = cv2.imread(iPath)
    vi = np.load(vPath).transpose(1,2,0)
    ims = [im, ttaug(im), ttaug(im), ttaug(im), ttaug(im)]
    vis = [vi] * len(ims)
    preds = []
    for im, vi in zip(ims, vis):
        with torch.no_grad():
            im = T.Compose([T.ToPILImage(),T.ToTensor()])(im).float()
            vi = T.Compose([T.ToTensor()])(vi).float()
            im = torch.unsqueeze(im, 0)
            vi = torch.unsqueeze(vi, 0)
            im = im.to(device)
            vi = vi.to(device)
            y_pred = model(im, vi)
            label = y_pred.cuda().data.cpu().numpy()
            if b_debug:
                print('[INFO] label:\n{}'.format(label))
            preds.append(label)
    if b_debug:
        print('[INFO] preds:\n{}'.format(preds))
    preds = np.mean(preds, axis=0)
    if b_debug:
        print('[INFO] mean preds:\n{}'.format(preds))
    preds = np.argmax(preds)
    if b_debug:
        print('[INFO] argmax mean preds:\n{}'.format(preds))
    return preds + 1


def dotta(model, b_debug=False):
    model.to(device)
    model.eval()
    testImagePaths = list(paths.list_images(config.test_data))
    testImagePaths.sort()
    testVisitPath = config.test_vis
    fo = open('submit/tta.txt', 'w')
    for iPath in testImagePaths:
        iName = iPath.split(os.path.sep)[-1]
        vName = iName.replace('jpg', 'npy')
        vPath = os.path.sep.join([testVisitPath, vName])
        preds = ttacore(model, iPath, vPath, b_debug)
        print('[INFO] ID-->Preds: {0} --> {1}'.format(iName[:-4].zfill(6), preds))
        line = iName[:-4].zfill(6) + '\t' + str(preds).zfill(3) + '\n'
        fo.write(line)
        if b_debug:
            break
    fo.close()
# ---------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    #main()
    fold = 0
    model=MultiModalNet("se_resnext50_32x4d","dpn26",0.5)
    best_model = torch.load("%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models,config.model_name,str(fold)))
    model.load_state_dict(best_model["state_dict"])
    dotta(model, False)
