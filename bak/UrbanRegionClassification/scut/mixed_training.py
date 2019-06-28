# USAGE
from scut import datasets
from scut import models
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from keras import losses
from keras.optimizers import Adam, RMSprop
from keras.layers import concatenate
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to tfrecord file for train and valid.")
args = vars(ap.parse_args())

dataFolder = args['dataset']

print("[INFO] load data from tfrecord file...")
trainVisitX, trainImageX, trainY = datasets.load_data(dataFolder, 'train.tfrecord')
testVisitX, testImageX, testY = datasets.load_data(dataFolder, 'valid.tfrecord')
# print('trainVisitX.shape:', trainVisitX.shape)
# print('trainImageX.shape:', trainImageX.shape)
# print('trainY.shape:', trainY.shape)

# print('testVisitX.shape:', testVisitX.shape)
# print('testImageX.shape:', testImageX.shape)
# print('testY.shape:', testY.shape)

print("[INFO] create and compile visit&image model...")
EPOCH = 20000
BATCH_SIZE = datasets.BATCH_SIZE
SUM_OF_ALL_DATASAMPLES = trainVisitX.shape[0]
STEPS_PER_EPOCH= SUM_OF_ALL_DATASAMPLES / BATCH_SIZE

# create the MLP and CNN models
visitModel = models.createVisitModel(32, 32, 7)
imageModel = models.createImageModel(64, 64, 3)

# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = concatenate([visitModel.output, imageModel.output])

# our final FC layer head will have two dense layers, the final one
# being our regression head 18->9
x = Dense(9, activation="softmax")(combinedInput)

# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (the
# predicted area code of the area)
model = Model(inputs=[visitModel.input, imageModel.input], outputs=x)

# initialize the number of epochs to train for, base learning rate,
# and batch size
# opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(
	optimizer=RMSprop(lr=0.0001),
	loss=losses.sparse_categorical_crossentropy,
	metrics=["acc"])

# train the model
print("[INFO] train visit&image model...")
# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(
	[trainVisitX, trainImageX], 
	trainY,
	validation_data=([testVisitX, testImageX], testY),
	epochs=EPOCH,
	batch_size=BATCH_SIZE,
	steps_per_epoch=STEPS_PER_EPOCH,
	callbacks=callbacks_list, verbose=0)
model.save("visitImage.h5")

# evaluate
print("[INFO] evaluate visit&image network...")
print(model.evaluate(x=[testVisitX, testImageX], y=testY))
