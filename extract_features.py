# -*- encoding:utf-8 -*-
# USAGE
# python extract_features.py

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from scut import config
from imutils import paths
import numpy as np
import pickle
import random
import os

# load the ResNet50 network and initialize the label encoder
print("[INFO] loading network...")
# model = ResNet50(weights="imagenet", include_top=False)
model = ResNet50(weights=None, include_top=False)
le = None

# loop over the data splits
for split in (config.TRAIN, config.TEST, config.VAL):
	# grab all file paths in the current split
	print("[INFO] processing '{} split'...".format(split))
	p = os.path.sep.join([config.BASE_PATH, split])
	filePaths = list(paths.list_files(p))

	# randomly shuffle the image paths and then extract the class
	# labels from the file paths
	random.shuffle(filePaths)
	labels = [p.split(os.path.sep)[-2] for p in filePaths]

	# if the label encoder is None, create it
	if le is None:
		le = LabelEncoder()
		le.fit(labels)

	# open the output CSV file for writing
	csvPath = os.path.sep.join([config.BASE_CSV_PATH,
		"{}.csv".format(split)])
	csv = open(csvPath, "w")

	# loop over the images in batches
	for (b, i) in enumerate(range(0, len(filePaths), config.BATCH_SIZE)):
		# extract the batch of images and labels, then initialize the
		# list of actual images that will be passed through the network
		# for feature extraction
		print("[INFO] processing batch {}/{}".format(b + 1,
			int(np.ceil(len(filePaths) / float(config.BATCH_SIZE)))))
		batchPaths = filePaths[i:i + config.BATCH_SIZE]
		batchLabels = le.transform(labels[i:i + config.BATCH_SIZE])
		batchFiles = []

		# loop over the images and labels in the current batch
		for filePath in batchPaths:
			file = np.load(filePath)
			file = np.expand_dims(file, axis=0)
			file = imagenet_utils.preprocess_input(file)
			# print(file.shape)
			# add the image to the batch
			batchFiles.append(file)

		# pass the images through the network and use the outputs as
		# our actual features, then reshape the features into a
		# flattened volume
		batchFiles = np.vstack(batchFiles)
		# print(batchFiles.shape)
		features = model.predict(batchFiles, batch_size=config.BATCH_SIZE)
		# print(features.shape) # 32, 3, 3, 2048
		# features = features.reshape((features.shape[0], 7 * 7 * 2048))
		features = features.reshape((features.shape[0], 3 * 3 * 2048))

		# loop over the class labels and extracted features
		for (label, vec) in zip(batchLabels, features):
			# construct a row that exists of the class label and
			# extracted features
			vec = ",".join([str(v) for v in vec])
			csv.write("{},{}\n".format(label, vec))

	# close the CSV file
	csv.close()

# serialize the label encoder to disk
f = open(config.LE_PATH, "wb")
f.write(pickle.dumps(le))
f.close()