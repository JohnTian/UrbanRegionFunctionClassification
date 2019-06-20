# -*- encoding:utf-8 -*-
import os

# initialize the path to the *original* input directory of images and visits
ORIG_INPUT_DATASET = "UrbanRegionData/train"
ORIG_EVAL_DATASET = "UrbanRegionData/test_eval"

# initialize the base path to the *new* directory that will contain
# our images and visits after computing the training and testing split
BASE_PATH = "dataset"
BASE_IMAGE_TYPE = "image"
BASE_VISIT_TYPE = "visit"

# define the names of the training, testing, and validation
# directories
TRAIN = "training"
TEST = "evaluation"
VAL = "validation"

# initialize the list of class label names
CLASSES = [
    "Residential area",
    "School",
    "Industrial park",
    "Railway station",
    "Airport",
    "Park",
    "Shopping area",
    "Administrative district",
    "Hospital"
]

# set the batch size
BATCH_SIZE = 512
EPOCH = 20

# set the path to the serialized model after training
MODEL_PATH = "saved_models"
MODEL_NAME = "urbanRegion.h5"
WARMUP_PLOT_PATH = os.path.sep.join([MODEL_PATH, "warmup.png"])
LOSS_PLOT_PATH = os.path.sep.join([MODEL_PATH, "loss.png"])
ACC_PLOT_PATH = os.path.sep.join([MODEL_PATH, "acc.png"])
