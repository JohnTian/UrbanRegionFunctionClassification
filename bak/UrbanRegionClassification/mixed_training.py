# -*- encodig:utf-8 -*-
import os
import keras
import numpy as np
from scut import resnet, models
from keras import Model
from keras.layers import Dense
from keras.layers import concatenate
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator


print("[INFO] load data ...")
trainVisitX = np.load('data/npy/trainVisit.npy')
trainImageX = np.load('data/npy/trainImage.npy')
trainY = np.load('data/npy/trainLabel.npy')

testVisitX = np.load('data/npy/validVisit.npy')
testImageX = np.load('data/npy/validImage.npy')
testY = np.load('data/npy/validLabel.npy')

print("[INFO] Normalize data...")
trainVisitX = trainVisitX.astype('float32') / np.max(trainVisitX)
trainImageX = trainImageX.astype('float32') / 255.0
testVisitX  = testVisitX.astype('float32') / np.max(testVisitX)
testImageX  = testImageX.astype('float32') / 255.0

# Convert class vectors to binary class matrices.
num_classes = 9
trainY = np.reshape(trainY, (trainY.shape[0], 1)) - 1
testY = np.reshape(testY, (testY.shape[0], 1)) - 1
trainY = keras.utils.to_categorical(trainY, num_classes)
testY = keras.utils.to_categorical(testY, num_classes)

print("[INFO] create and compile visit&image model...")
n = 3
version = 1
# visitModel = resnet.create_resnet(input_shape=trainVisitX.shape[1:], n=n, version=version, num_classes=num_classes)
visitModel = models.create_cnn(32, 32, 7)
imageModel = resnet.create_resnet(input_shape=trainImageX.shape[1:], n=n, version=version, num_classes=num_classes)
combinedInput = concatenate([visitModel.output, imageModel.output])
x = Dense(9, activation="softmax")(combinedInput)
model = Model(inputs=[visitModel.input, imageModel.input], outputs=x)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=resnet.lr_schedule(0)),
              metrics=['accuracy'])
model.summary()


print("[INFO] prepare saving directory and checkpoint visit&image model...")
## Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2
model_type = 'ResNet%dv%d' % (depth, version)
model_name = 'urban_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

## Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)
lr_scheduler = LearningRateScheduler(resnet.lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
callbacks = [checkpoint, lr_reducer, lr_scheduler]


print("[INFO] training visit&image model...")
data_augmentation = True
EPOCH = 200
BATCH_SIZE = 32 # 128
SUM_OF_ALL_DATASAMPLES = trainVisitX.shape[0]
STEPS_PER_EPOCH= SUM_OF_ALL_DATASAMPLES / BATCH_SIZE

model.fit([trainVisitX, trainImageX], trainY,
            batch_size=BATCH_SIZE,
            epochs=EPOCH,
            validation_data=([testVisitX, testImageX], testY),
            shuffle=True,
            callbacks=callbacks)

print("[INFO] evaluating visit&image model...")
# Score trained model.
scores = model.evaluate([testVisitX, testImageX], testY, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
