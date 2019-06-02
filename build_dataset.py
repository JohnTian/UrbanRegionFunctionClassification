# -*- encoding:utf-8 -*-
# USAGE
# python build_dataset.py

# import the necessary packages
import os
import cv2
import shutil
import random
import datetime
import numpy as np
import pandas as pd
from scut import config
from imutils import paths
from collections import OrderedDict


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

def visit2array(table):
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
    return init / np.max(init)


def random_crop_and_normal(image_path, h=88, w=88):
    im = cv2.imread(image_path)
    height, width, _ = im.shape
    y = random.randint(1, height - h)
    x = random.randint(1, width - w)
    crop = im[y:y+h, x:x+w] / 255.0
    return crop


def build_DataSet(dataFolder):
	files = OrderedDict()
	for code in config.CLASSES:
		files[code] = []
	for filePath in paths.list_files(dataFolder):
		if filePath[-3:] == 'jpg':
			# UrbanRegionData/train/007/030516_007.jpg
			filename = filePath.split(os.path.sep)[-1]
			label = config.CLASSES[int(filename.split('_')[-1].split('.')[0])-1]
			files[label].append(filePath)
	minNum = min([len(v) for v in files.values()])
	testNum = int(np.ceil(minNum*0.1))
	validNum = testNum
	trainData, testData, validData = {}, {}, {}
	for label in files.keys():
		testData[label] = random.sample(files[label], testNum)
		validData[label] = random.sample(set(files[label]) - set(testData[label]), validNum)
		trainData[label] = list(set(files[label]) - set(testData[label]) - set(validData[label]))
	
	# loop over the data splits
	for split, data in zip((config.TRAIN, config.TEST, config.VAL), (trainData, testData, validData)):
		# grab all image paths in the current split
		print("[INFO] processing '{} split'...".format(split))
		for label, filePaths in data.items():
			# construct the path to the output directory
			dirPath = os.path.sep.join([config.BASE_PATH, split, label])
			# if the output directory does not exist, create it
			if not os.path.exists(dirPath):
				os.makedirs(dirPath)
			for filePath in filePaths:
				# image 随机裁剪加归一化 88*88*3
				image = random_crop_and_normal(filePath)
				image = np.reshape(image, (1, image.shape[0]*image.shape[1]*image.shape[2]))
				# visit 归一化 7x26x24
				table = pd.read_table(filePath.replace('jpg','txt'), header=None)
				visit = visit2array(table)
				visit = np.reshape(visit, (1, visit.shape[0]*visit.shape[1]*visit.shape[2]))
				# merge 88*88*3 + 7*26*24 = 27600  96*96*3 = 27648 相差48
				imiv = np.append(image, visit)
				imiv = np.append(imiv, [0]*48)
				imiv = np.reshape(imiv, (96, 96, 3))
				# save
				filename = filePath.split(os.path.sep)[-1]
				p = os.path.sep.join([dirPath, filename.replace('jpg', 'npy')])
				np.save(p, imiv)
		print('[INFO] {} done!'.format(split))


if __name__ == "__main__":
	build_DataSet(config.ORIG_INPUT_DATASET)