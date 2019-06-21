# -*- encoding:utf-8 -*-
# USAGE: python train.py
import os
import keras
import numpy as np
from keras import optimizers
from imutils import paths
from scut import config
from scut.util import plot_training_loss
from scut.data import preprocessing_data_gen, image_gen
from scut.model.models import create_image_model, lr_schedule, create_image_model_baseon_pretrained
from sklearn.metrics import classification_report


print("[INFO] loading data ...")
data = preprocessing_data_gen()
totalTrain = data['totalTrain']
totalVal = data['totalVal']
totalTest = data['totalTest']
testLabels = data['testLabels']
trainImagePath = data['trainImagePath']
validImagePath = data['validImagePath']
testImagePath = data['testImagePath']
print('[INFO] totalTrain: {}, totalVal: {}, totalTest: {}'.format(totalTrain, totalVal, totalTest))

print("[INFO] creating generate object ...")
trainGen = image_gen(trainImagePath, 'train')
validGen = image_gen(validImagePath, 'valid')
testGen = image_gen(testImagePath, 'test')

print("[INFO] building model ...")
model = create_image_model(88, 88, 3)
#model = create_image_model_baseon_pretrained(88, 88, 3)
model.summary()
# save model structure in file
if not os.path.exists(config.MODEL_PATH):
    os.makedirs(config.MODEL_PATH)
modelStructurePath = os.path.sep.join([config.MODEL_PATH, 'urbanRegionImage.png'])
keras.utils.plot_model(
	model,
	show_shapes=True,
	show_layer_names=True,
	to_file=modelStructurePath
)

print("[INFO] compiling model ...")
#model.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=["accuracy"])
#model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=["accuracy"])
model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adamax(), metrics=["accuracy"])

print("[INFO] config callbacks ...")
# Prepare model model saving directory.
save_dir = config.MODEL_PATH
if not os.path.isdir(save_dir):
	os.makedirs(save_dir)
filepath = os.path.join(save_dir, config.MODEL_NAME)
# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = keras.callbacks.ModelCheckpoint(
	filepath=filepath,
	monitor='val_acc',
	verbose=1,
	save_best_only=True
)
lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
lr_reducer = keras.callbacks.ReduceLROnPlateau(
	factor=0.1,
	patience=10,
	min_lr=1e-6
)
callbacks = [
	checkpoint,
	lr_reducer,
	# lr_scheduler,
	keras.callbacks.TensorBoard(log_dir='log',histogram_freq=0)
]

print("[INFO] training model ...")
H = model.fit_generator(
        trainGen,
	    steps_per_epoch=totalTrain // config.BATCH_SIZE,
        validation_data=validGen,
        validation_steps=totalVal // config.BATCH_SIZE,
        epochs=config.EPOCH,
        callbacks=callbacks)

print("[INFO] evaluating model by predict ...")
predIdxs = model.predict_generator(
	testGen,
	steps=(totalTest // config.BATCH_SIZE) + 1
)
predIdxs = np.argmax(predIdxs, axis=1)
print(
	classification_report(
		testLabels,
		predIdxs,
		target_names=config.CLASSES
	)
)

print("[INFO] plot image for training ...")
plot_training_loss(H, config.EPOCH, config.LOSS_PLOT_PATH)
