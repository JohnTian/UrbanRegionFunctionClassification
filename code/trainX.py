# -*- encoding:utf-8 -*-
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.python import keras as keras

batch_size = 512
epochs = 200
num_classes = 9

def read_and_decode_train(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'data': tf.FixedLenFeature([], tf.string),
                                           'visit': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                       })
    image = tf.decode_raw(features['data'], tf.uint8)
    image = tf.reshape(image, [100, 100, 3])
    image = tf.random_crop(image, [88, 88, 3])
    image = tf.cast(image, tf.float32) / 255.0

    visit = tf.decode_raw(features['visit'], tf.float32)
    visit = tf.reshape(visit, [7, 26, 24])
    visit = tf.transpose(visit, [2, 1, 0])
    visit = visit / tf.reduce_max(visit)
    label = tf.cast(features['label'], tf.int64)
    label = tf.one_hot(label, num_classes)
    return image, visit, label

def read_and_decode_valid(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'data': tf.FixedLenFeature([], tf.string),
                                           'visit': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                       })
    image = tf.decode_raw(features['data'], tf.uint8)
    image = tf.reshape(image, [100, 100, 3])
    image = tf.random_crop(image, [88, 88, 3])
    image = tf.cast(image, tf.float32)/255.0

    visit = tf.decode_raw(features['visit'], tf.float32)
    visit = tf.reshape(visit, [7, 26, 24])
    visit = tf.transpose(visit, [2, 1, 0])
    visit = visit / tf.reduce_max(visit)
    label = tf.cast(features['label'], tf.int64)
    label = tf.one_hot(label, num_classes)
    return image, visit, label

def load_training_set():
    with tf.name_scope('input_train'):
        image_train, visit_train, label_train = read_and_decode_train("../data/tfrecord/train.tfrecord")
        image_batch_train, visit_batch_train, label_batch_train = tf.train.shuffle_batch(
            [image_train, visit_train, label_train], batch_size=batch_size, capacity=2048, min_after_dequeue=2000, num_threads=4
        )
    return image_batch_train, visit_batch_train, label_batch_train

def load_valid_set():
    with tf.name_scope('input_valid'):
        image_valid, visit_valid, label_valid = read_and_decode_valid("../data/tfrecord/valid.tfrecord")
        image_batch_valid, visit_batch_valid, label_batch_valid = tf.train.shuffle_batch(
            [image_valid, visit_valid, label_valid], batch_size=batch_size, capacity=2048, min_after_dequeue=2000, num_threads=4
        )
    return image_batch_valid, visit_batch_valid, label_batch_valid

def build_image_model(height, width, channel):
    input_shape = (height, width, channel)
    image_base_model = keras.applications.Xception(
        include_top=False,
        weights='imagenet',
        #input_tensor=image_input,
        input_shape=input_shape)
    image_base_model.trainable = False
    image_x = image_base_model.output
    image_x = keras.layers.GlobalAveragePooling2D()(image_x)
    image_x = keras.layers.Dense(1024, activation='relu')(image_x)
    image_y = keras.layers.Dense(nb_classes, activation='relu')(image_x)
    image_model = keras.models.Model(inputs=image_base_model.input, outputs=image_y, name='image')
    return image_model

def build_visit_model(visit_input, filters=(16, 32, 64)):
    chanDim = -1

    # define the model input
    inputs = visit_input

    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs

        # CONV => RELU => BN => POOL
        x = keras.layers.Conv2D(f, (3, 3), padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.BatchNormalization(axis=chanDim)(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(16)(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.BatchNormalization(axis=chanDim)(x)
    x = keras.layers.Dropout(0.5)(x)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = keras.layers.Dense(9)(x)
    x = keras.layers.Activation("relu")(x)

    # construct the CNN
    visit_model = keras.models.Model(inputs, x)
    return visit_model


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def train(nb_classes):
    ## Load data
    image_batch_train, visit_batch_train, label_batch_train = load_training_set()
    image_batch_valid, visia_batch_valid, label_batch_valid = load_valid_set()

    ## Build model
    image_model = build_image_model(88, 88, 3)

    visit_input = keras.layers.Input(shape=(26,24,7))
    visit_model = build_visit_model(visit_input)
    combined_x = keras.layers.concatenate([image_model.output, visit_model.output])
    output = keras.layers.Dense(nb_classes, activation="softmax")(combined_x)
    model = keras.models.Model(
        inputs=[image_model.input, visit_model.input],
        outputs=output,
        name='imageVisit'
    )
    model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),
                metrics=['accuracy'])
    model.summary()

    ## Fit and Train
    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'urbanRegion_%s_model.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True)

    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
    lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                                cooldown=0,
                                patience=5,
                                min_lr=0.5e-6)
    callbacks = [checkpoint, lr_reducer, lr_scheduler]
    # Fit
    model.fit(
        x=[image_batch_train, visit_batch_train],
        y=[label_batch_train],
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([image_batch_valid, visia_batch_valid], label_batch_valid),
        shuffle=True,
        callbacks=callbacks)

    ## Score trained model
    scores = model.evaluate([image_batch_valid, visia_batch_valid], label_batch_valid, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


if __name__ == "__main__":
    nb_classes = 9
    train(nb_classes)
