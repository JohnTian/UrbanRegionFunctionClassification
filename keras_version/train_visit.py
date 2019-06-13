# -*- encoding:utf-8 -*-
# USAGE: python train.py
import os
import keras
import numpy as np
from imutils import paths
from scut import config
from scut.util import plot_training
from scut.data import preprocessing_data_gen, visit_gen
from scut.models import create_image_model, create_visit_model, lr_schedule
from keras.models import Model
from keras.optimizers import SGD
from keras.layers.core import Dense
from sklearn.metrics import classification_report


print("[INFO] loading data ...")
data = preprocessing_data_gen()
totalTrain = data['totalTrain']
totalVal = data['totalVal']
totalTest = data['totalTest']
testLabels = data['testLabels']
trainVisitPath = data['trainVisitPath']
validVisitPath = data['validVisitPath']
testVisitPath = data['testVisitPath']
print('[INFO] totalTrain: {}, totalVal: {}, totalTest: {}'.format(totalTrain, totalVal, totalTest))

print("[INFO] creating generate object ...")
trainGen = visit_gen(trainVisitPath, 'train')
validGen = visit_gen(validVisitPath, 'valid')
testGen = visit_gen(testVisitPath, 'test')

print("[INFO] building model ...")
model = create_visit_model(24, 26, 7)
model.summary()
keras.utils.plot_model(
	model,
	show_shapes=True,
	show_layer_names=True,
	to_file='output/urbanRegionVisit.png'
)

print("[INFO] compiling model ...")
# opt = SGD(lr=1e-4, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=["accuracy"])

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
	factor=np.sqrt(0.1),
	cooldown=0,
	patience=5,
	min_lr=0.5e-6
)
callbacks = [
	checkpoint,
	#lr_reducer,
	#lr_scheduler,
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
plot_training(H, config.EPOCH, config.WARMUP_PLOT_PATH)
