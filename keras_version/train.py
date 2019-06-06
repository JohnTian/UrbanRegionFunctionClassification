# -*- encoding:utf-8 -*-
# USAGE: python train.py
import keras
import numpy as np
from scut import config
from scut.util import plot_training
from scut.data import create_image_gen, create_visit_gen, load_image_data, load_visit_data
from scut.models import create_image_model, create_visit_model, lr_schedule
from keras.models import Model
from keras.optimizers import SGD
from keras.layers.core import Dense
from sklearn.metrics import classification_report


print("[INFO] loading data ...")
(trainImageData, trainImageLabel), (validImageData, validImageLabel), (testImageData, testImageLabel) = load_image_data()
(trainVisitData, trainVisitLabel), (validVisitData, validVisitLabel), (testVisitData, testVisitLabel) = load_visit_data()
totalTrain = trainImageLabel.shape[0]
totalVal = validImageLabel.shape[0]
totalTest = testImageLabel.shape[0]

# Convert class vectors to binary class matrices.
num_classes = 9
trainY = keras.utils.to_categorical(trainImageLabel, num_classes)
validY = keras.utils.to_categorical(validImageLabel, num_classes)
testY = keras.utils.to_categorical(testImageLabel, num_classes)

print("[INFO] building model ...")
imageModel = create_image_model(100, 100, 3)
visitModel = create_visit_model(26, 24, 7)
combinedInput = concatenate([imageModel.output, visitModel.output])
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
model_name = 'urbanRegion_%s_model.h5'
filepath = os.path.join(save_dir, model_name)
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
H = model.fit(
	[trainImageData, trainVisitData],
	trainY,
	steps_per_epoch=totalTrain // config.BATCH_SIZE,
	validation_data=([validImageData, validVisitData], validY),
	validation_steps=totalVal // config.BATCH_SIZE,
	epochs=50,
	callbacks=callbacks)

print("[INFO] evaluating after fine-tuning model ...")
predIdxs = model.predict(
	[testImageData, testVisitData],
	steps=(totalTest // config.BATCH_SIZE) + 1
)
predIdxs = np.argmax(predIdxs, axis=1)
print(
	classification_report(
		testY,
		predIdxs,
		target_names=testImageLabel
	)
)

print("[INFO] plot image for training ...")
plot_training(H, 50, config.WARMUP_PLOT_PATH)
