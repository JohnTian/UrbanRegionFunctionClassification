# -*- encoding:utf-8 -*-
# USAGE: python build_dataset.py
# import the necessary packages
import os
import shutil
import random
import numpy as np
import pandas as pd
from scut import config
from imutils import paths
from scut.util import randomCropAndNormal
from scut.util import visit2arrayAndNormal
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
		trainData[label] = list(set(files[label]) - set(testData[label]) - set(validData[label]))
	
	# loop over the data splits
	for split, data in zip((config.TRAIN, config.TEST, config.VAL), (trainData, testData, validData)):
		# grab all image paths in the current split
		print("[INFO] processing '{} split'...".format(split))

		for label, filePaths in data.items():
			# construct the path to the output directory for image and visit
			dirImagePath = os.path.sep.join([config.BASE_PATH, 'image', split, label])
			dirVisitPath = os.path.sep.join([config.BASE_PATH, 'visit', split, label])
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


if __name__ == "__main__":
	build_DataSet(config.ORIG_INPUT_DATASET)