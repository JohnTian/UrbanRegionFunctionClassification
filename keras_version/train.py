# -*- encoding:utf-8 -*-
# USAGE: python train.py
import os
import keras
import numpy as np
from imutils import paths
from scut import config
from scut.util import plot_training
from scut.data import create_data_gen
from scut.models import create_image_model, create_visit_model, lr_schedule
from keras.models import Model
from keras.optimizers import SGD
from keras.layers.core import Dense
from sklearn.metrics import classification_report


print("[INFO] loading data ...")
trainImagePath = os.path.sep.join([config.BASE_PATH, config.BASE_IMAGE_TYPE, config.TRAIN])
validImagePath = os.path.sep.join([config.BASE_PATH, config.BASE_IMAGE_TYPE, config.VAL])
testImagePath = os.path.sep.join([config.BASE_PATH, config.BASE_IMAGE_TYPE, config.TEST])

trainVisitPath = os.path.sep.join([config.BASE_PATH, config.BASE_VISIT_TYPE, config.TRAIN])
validVisitPath = os.path.sep.join([config.BASE_PATH, config.BASE_VISIT_TYPE, config.VAL])
testVisitPath = os.path.sep.join([config.BASE_PATH, config.BASE_VISIT_TYPE, config.TEST])

totalTrain = len(list(paths.list_images(trainImagePath)))
totalVal = len(list(paths.list_images(validImagePath)))

testFiles = list(paths.list_images(testImagePath))
testLabels = [config.CLASSES.index(line.split(os.path.sep)[-2]) for line in testFiles]
testNames = [line.split(os.path.sep)[-2] for line in testFiles]
totalTest = len(testLabels)

trainGen = create_data_gen(trainImagePath, trainVisitPath, 'train')
validGen = create_data_gen(validImagePath, validVisitPath, 'valid')
testGen = create_data_gen(testImagePath, testVisitPath, 'test')

print("[INFO] building model ...")
imageModel = create_image_model(100, 100, 3)
visitModel = create_visit_model(32, 32, 7)
combinedInput = keras.layers.concatenate([imageModel.output, visitModel.output])
x = Dense(16, activation="relu")(combinedInput)
x = Dense(9, activation="softmax")(x)
model = Model(inputs=[imageModel.input, visitModel.input], outputs=x)

print("[INFO] compiling model ...")
opt = SGD(lr=1e-4, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

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
	lr_reducer,
	lr_scheduler,
	keras.callbacks.TensorBoard(
        log_dir='log',
        histogram_freq=0
    )
]

print("[INFO] training model ...")
model.fit_generator(
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
		target_names=testNames
	)
)

print("[INFO] plot image for training ...")
plot_training(H, 50, config.WARMUP_PLOT_PATH)
