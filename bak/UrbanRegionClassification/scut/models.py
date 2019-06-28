# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.applications import densenet, ResNet50, VGG19


def create_mlp(dim, regress=False):
	# define our MLP network
	model = Sequential()
	model.add(Dense(16, input_dim=dim, activation="relu")) # 8
	model.add(Dense(9, activation="relu")) # 4

	# check to see if the regression node should be added
	if regress:
		model.add(Dense(1, activation="linear"))

	# return our model
	return model

def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1

	# define the model input
	inputs = Input(shape=inputShape)

	# loop over the number of filters
	for (i, f) in enumerate(filters):
		# if this is the first CONV layer then set the input
		# appropriately
		if i == 0:
			x = inputs

		# CONV => RELU => BN => POOL
		x = Conv2D(f, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

	# flatten the volume, then FC => RELU => BN => DROPOUT
	x = Flatten()(x)
	x = Dense(16)(x)
	x = Activation("relu")(x)
	x = BatchNormalization(axis=chanDim)(x)
	x = Dropout(0.5)(x)

	# apply another FC layer, this one to match the number of nodes
	# coming out of the MLP
	x = Dense(9)(x)
	x = Activation("relu")(x)

	# construct the CNN
	model = Model(inputs, x)

	# return the CNN
	return model


def createVisitModel(height, width, depth):
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	# define the model input
	inputs = Input(shape=inputShape)
	return ResNet50(
		include_top=True,
		weights=None,
		input_tensor=inputs,
		input_shape=inputShape,
		pooling=None,
		classes=9)

def createImageModel(height, width, depth):
	inputShape = (height, width, depth)
	# define the model input
	inputs = Input(shape=inputShape)
    # at least 32x32
	return densenet.DenseNet201(include_top=True,
		weights=None,
		input_tensor=inputs,
		input_shape=inputShape,
		pooling=None,
		classes=9)