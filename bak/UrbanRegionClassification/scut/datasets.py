# -*- encoding:utf-8 -*-
# import the necessary packages
import os
import cv2
import time
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def load_visit_attributes(inputPath):
	# initialize the list of column names in the CSV file and then
	# load it using Pandas
	cols = [
		'id', 'area',
		'0h', '1h', '2h', '3h', '4h', '5h', '6h', '7h', 
		'8h', '9h', '10h', '11h', '12h', '13h', '14h', '15h', 
		'16h', '17h', '18h', '19h', '20h', '21h', '22h', '23h',
		'1d', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', '10d',
		'11d', '12d', '13d', '14d', '15d', '16d', '17d', '18d', '19d', '20d',
		'21d', '22d', '23d', '24d', '25d', '26d', '27d', '28d', '29d', '30d', '31d',
		'0w', '1w', '2w', '3w', '4w', '5w', '6w',
		'1m', '2m', '3m', '10m', '11m', '12m'
	]
	df = pd.read_csv(inputPath, sep=",", header=None, names=cols)
	# drop rows which contain 'null'
	for colname in cols[2:]:
		idxs = df[df[colname] == "null"].index.tolist()
		df.drop(idxs, inplace=True)
		# df[colname][df[colname]=='null'] = 0
	
	## 筛选各个类中最小的样本个数
	minNum = np.inf
	classes = max(df['area']) + 1
	for i in range(classes):
		d = df[df['area']==i]
		l = len(d)
		if minNum > l:
			minNum = l
	dfF = pd.DataFrame()
	for i in range(classes):
		d = df[df['area']==i][:minNum]
		dfF = dfF.append(d)

	return dfF

def process_visit_attributes(df, train, test):
	# initialize the column names of the continuous data
	# continuous = ['0h', '1h', '2h', '3h', '4h', '5h', '6h', '7h', 
	# 	'8h', '9h', '10h', '11h', '12h', '13h', '14h', '15h', 
	# 	'16h', '17h', '18h', '19h', '20h', '21h', '22h', '23h',
	# 	'1d', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', '10d',
	# 	'11d', '12d', '13d', '14d', '15d', '16d', '17d', '18d', '19d', '20d',
	# 	'21d', '22d', '23d', '24d', '25d', '26d', '27d', '28d', '29d', '30d', '31d',
	# 	'0w', '1w', '2w', '3w', '4w', '5w', '6w',
	# 	'1m', '2m', '3m', '10m', '11m', '12m'
	# ]
	continuous = ['0w', '1w', '2w', '3w', '4w', '5w', '6w']

	# performin min-max scaling each continuous feature column to
	# the range [0, 1]
	cs = MinMaxScaler()
	trainX = cs.fit_transform(train[continuous])
	testX = cs.transform(test[continuous])

	# return the concatenated training and testing data
	return (trainX, testX)

#----------------------------------------------------------
SHUFFLE_BUFFER = 500
BATCH_SIZE = 128
NUM_CLASSES = 9
def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {
		'visit': tf.FixedLenFeature([], tf.string),
		'image': tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64)
	}
    # Load one example
    parsed_features = tf.parse_single_example(proto, keys_to_features)
    # Turn your saved image string into an array
    parsed_features['visit'] = tf.decode_raw(parsed_features['visit'], tf.float32)
    parsed_features['image'] = tf.decode_raw(parsed_features['image'], tf.uint8)
    return parsed_features['visit'], parsed_features['image'], parsed_features["label"]

def create_dataset(filepath):
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8)
    # This dataset will go on forever
    dataset = dataset.repeat()
    # Set the number of datapoints you want to load and shuffle 
    dataset = dataset.shuffle(SHUFFLE_BUFFER)
    # Set the batchsize
    dataset = dataset.batch(BATCH_SIZE)
    # Create an iterator
    iterator = dataset.make_one_shot_iterator()
    # Create your tf representation of the iterator
    visit, image, label = iterator.get_next()
    
	# Bring your picture back in shape
    visit = tf.reshape(visit, [-1, 32, 32, 7])
    visit = visit / np.max(visit)

    image = tf.reshape(image, [-1, 100, 100, 3])
    image = tf.cast(image, tf.float32) / 255.0
    
	# Create a one hot array for your labels
    # label = tf.one_hot(label, NUM_CLASSES)
    label = tf.cast(label, tf.float32)
    
    return visit, image, label

def load_data(tfRecordPath, dataname):
	tfPath = os.path.join(tfRecordPath, dataname)
	return create_dataset(tfPath)

