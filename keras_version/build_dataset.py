# -*- encoding:utf-8 -*-
# USAGE: python build_dataset.py
import os
import shutil
import random
import numpy as np
import pandas as pd
from scut import config
from imutils import paths
from scut.util import randomCropAndNormal
from scut.util import visit2arrayAndNormal
from scut.util import computeRatioOfBlackAndWhite
from collections import OrderedDict


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
		# trainData[label] = random.sample(set(files[label]) - set(testData[label]) - set(validData[label]), validNum*3)
		trainData[label] = list(set(files[label]) - set(testData[label]) - set(validData[label]))

	# loop over the data splits
	for split, data in zip((config.TRAIN, config.TEST, config.VAL), (trainData, testData, validData)):
		# grab all image paths in the current split
		print("[INFO] processing '{} split'...".format(split))

		for label, filePaths in data.items():
			# construct the path to the output directory for image and visit
			dirImagePath = os.path.sep.join([config.BASE_PATH, config.BASE_IMAGE_TYPE, split, label])
			dirVisitPath = os.path.sep.join([config.BASE_PATH, config.BASE_VISIT_TYPE, split, label])
			if not os.path.exists(dirImagePath):
				os.makedirs(dirImagePath)
			if not os.path.exists(dirVisitPath):
				os.makedirs(dirVisitPath)

			for filePath in filePaths:
				# ## Method 1 just copy
				# filename = filePath.split(os.path.sep)[-1]
				# pImagePath = os.path.sep.join([dirImagePath, filename])
				# shutil.copy2(filePath, pImagePath)
				# pVisitPath = os.path.sep.join([dirVisitPath, filename.replace('jpg', 'npy')])
				# shutil.copy2(filePath.replace('jpg','txt'), pVisitPath)

				## Method 2 copy and visit2array
				filename = filePath.split(os.path.sep)[-1]
				pImagePath = os.path.sep.join([dirImagePath, filename])
				shutil.copy2(filePath, pImagePath)

				visit = visit2arrayAndNormal(filePath.replace('jpg','txt'))
				pVisitPath = os.path.sep.join([dirVisitPath, filename.replace('jpg', 'npy')])
				np.save(pVisitPath, visit)

				# ## Method 3 merge and normalize
				# # image
				# image = randomCropAndNormal(filePath)
				# image = np.reshape(image, (1, image.shape[0]*image.shape[1]*image.shape[2]))
				# # visit
				# visit = visit2arrayAndNormal(filePath.replace('jpg','txt'))
				# visit = np.reshape(visit, (1, visit.shape[0]*visit.shape[1]*visit.shape[2]))
				# # merge 88*88*3 + 7*26*24 = 27600  96*96*3 = 27648 相差48
				# image = np.append(image, [0]*48)
				# imiv = np.append(image, visit)
				# # normalize
				# imiv = imiv / np.max(imiv)
				# # reshape
				# imiv = np.reshape(imiv, (96, 96, 3))
				# # save
				# filename = filePath.split(os.path.sep)[-1]
				# p = os.path.sep.join([dirPath, filename.replace('jpg', 'npy')])
				# np.save(p, imiv)
		print('[INFO] {} done!'.format(split))


def build_filter_DataSet(dataFolder, bDebug=False):
	# 记录各个类中图像的地址
	files = OrderedDict()
	for code in config.CLASSES:
		files[code] = []

	print('[INFO] filter original data...')
	for filePath in paths.list_files(dataFolder):
		if filePath[-3:] == 'jpg':
			# UrbanRegionData/train/007/030516_007.jpg
			ratioOfBlack, ratioOfWhite = computeRatioOfBlackAndWhite(filePath)
			if (ratioOfBlack > 0.25) or (ratioOfWhite > 0.75):
				continue
			else:
				filename = filePath.split(os.path.sep)[-1]
				label = config.CLASSES[int(filename.split('_')[-1].split('.')[0])-1]
				files[label].append(filePath)
	if bDebug:
		print('[DEBUG] after filter original data...')
		for k, v in files.items():
			print('[DEBUG] after filter {:>25}:{:>5}'.format(k, len(v)))
		print('[DEBUG] after filter original data sum:', sum([len(v) for v in files.values()]))

	# 查找各个类中最小的图像个数 -- 各个类对应的图像个数不一样
	print('[INFO] split original data into testData, validData, trainData...')
	minNum = min([len(v) for v in files.values()])
	testNum = int(np.ceil(minNum*0.1))
	trainData, testData, validData = {}, {}, {}
	for label in files.keys():
		testData[label] = random.sample(files[label], testNum)
		validData[label] = random.sample(list(set(files[label]) - set(testData[label])), testNum)
		# trainData[label] = random.sample(set(files[label]) - set(testData[label]) - set(validData[label]), testNum*3)
		trainData[label] = list(set(files[label]) - set(testData[label]) - set(validData[label]))

	if bDebug:
		for k, v in testData.items():
			print('[DEBUG] testData {:>25}:{:>5}'.format(k, len(v)))
		print('[DEBUG] testData sum:', sum([len(v) for v in testData.values()]))

		for k, v in validData.items():
			print('[DEBUG] validData {:>25}:{:>5}'.format(k, len(v)))
		print('[DEBUG] validData sum:', sum([len(v) for v in validData.values()]))

		print('[DEBUG] before balance trainData...')
		for k, v in trainData.items():
			print('[DEBUG] before balance TrainData {:>25}:{:>5}'.format(k, len(v)))
		print('[DEBUG] before balance TrainData sum:', sum([len(v) for v in trainData.values()]))

	# loop over the data splits
	for split, data in zip((config.TRAIN, config.TEST, config.VAL), (trainData, testData, validData)):
	#for split, data in zip((config.TEST, config.VAL), (testData, validData)):
		print("[INFO] processing '{} split'...".format(split))
		for label, filePaths in data.items():
			# construct the output directory for image and visit
			dirImagePath = os.path.sep.join([config.BASE_PATH, config.BASE_IMAGE_TYPE, split, label])
			dirVisitPath = os.path.sep.join([config.BASE_PATH, config.BASE_VISIT_TYPE, split, label])
			if not os.path.exists(dirImagePath):
				os.makedirs(dirImagePath)
			if not os.path.exists(dirVisitPath):
				os.makedirs(dirVisitPath)
			# save image and visit
			for filePath in filePaths:
				# image
				filename = filePath.split(os.path.sep)[-1]
				pImagePath = os.path.sep.join([dirImagePath, filename])
				if not os.path.exists(pImagePath):
					shutil.copy2(filePath, pImagePath)
				# visit
				pVisitPath = os.path.sep.join([dirVisitPath, filename.replace('jpg', 'npy')])
				if not os.path.exists(pVisitPath):
					visit = visit2arrayAndNormal(filePath.replace('jpg','txt'))
					np.save(pVisitPath, visit)
		print('[INFO] {} done!'.format(split))

if __name__ == "__main__":
	build_filter_DataSet(config.ORIG_INPUT_DATASET, True)
