# -*- encoding:utf-8 -*-
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
import os
import cv2
import random
import datetime
import numpy as np
import pandas as pd
from imutils import paths
import matplotlib.pyplot as plt

def plot_training(H, N, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)

def plot_training_loss(H, N, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.title("Training Loss")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)

def plot_training_acc(H, N, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
	plt.title("Training Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)


# Refer repo: https://github.com/czczup/UrbanRegionFunctionClassification.git
# "00": 0, "01": 1, ..., "23": 23
str2int = {}
for i in range(24):
	str2int[str(i).zfill(2)] = i

# 访问记录内的时间从2018年10月1日起，共182天即26周, 每周7天, 将日期按日历排列
date2position = {}
datestr2dateint = {}
for i in range(182):
	date = datetime.date(day=1, month=10, year=2018)+datetime.timedelta(days=i)
	date_int = int(date.__str__().replace("-", ""))
	# 20181001: [0, 0]
	date2position[date_int] = [i%7, i//7]
	# "20181001": 20181001
	datestr2dateint[str(date_int)] = date_int

def visit2arrayAndNormal(filePath):
    table = pd.read_csv(filePath, header=None, sep='\t')
    strings = table[1]
    # 7天, 26周, 每天24小时
    init = np.zeros((7, 26, 24))
    for string in strings:
        temp = []
        for item in string.split(','):
            temp.append([item[0:8], item[9:].split("|")])
        for date, visit_lst in temp:
            # 第y周的第x天的到访总人数为value
            x, y = date2position[datestr2dateint[date]]
            for visit in visit_lst:
                init[x][y][str2int[visit]] += 1
    init = init / np.max(init)
    init = np.transpose(init, [2, 1, 0]) # (7, 26, 24) --> (24, 26, 7)
    return init


def randomCropAndNormal(image_path, h=88, w=88):
    im = cv2.imread(image_path)
    height, width, _ = im.shape
    y = random.randint(1, height - h)
    x = random.randint(1, width - w)
    crop = im[y:y+h, x:x+w]
    return crop / 255.0

def save2txt(pathOfData, validExts, pathOfSaveTxt):
    """
    save path of data to txt file.
    """
    filePaths = paths.list_files(pathOfData, validExts=validExts)
    if os.path.exists(pathOfSaveTxt):
        os.remove(pathOfSaveTxt)
        #print('[INFO] {} removed'.format(pathOfSaveTxt))
    with open(pathOfSaveTxt, 'w+') as fo:
        for fPath in filePaths:
            fo.write(fPath+'\n')


def computeRatioOfBlackAndWhite(imgPath):
    im = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    imHeight, imWidth = np.shape(im)
    imSize = imHeight * imWidth
    hist = cv2.calcHist([im],[0],None,[256],[0,256])
    ratioOfBlack = 1.0 * hist[0,0] / imSize
    ratioOfWhite = 1.0 * hist[255,0] / imSize
    return (ratioOfBlack, ratioOfWhite)