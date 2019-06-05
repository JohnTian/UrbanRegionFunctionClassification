# -*- encoding:utf-8 -*-
# import the necessary packages
import os

# initialize the path to the *original* input directory of images and visits
ORIG_INPUT_DATASET = "UrbanRegionData/train"

# initialize the base path to the *new* directory that will contain
# our images and visits after computing the training and testing split
BASE_PATH = "dataset"

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
BATCH_SIZE = 64

# initialize the label encoder file path and the output directory to
# where the extracted features (in CSV file format) will be stored
LE_PATH = os.path.sep.join(["output", "le.cpickle"])
BASE_CSV_PATH = "output"

# set the path to the serialized model after training
MODEL_PATH = os.path.sep.join(["output", "urbanRegion.model"])

# define the path to the output training history plots
UNFROZEN_PLOT_PATH = os.path.sep.join(["output", "unfrozen.png"])
WARMUP_PLOT_PATH = os.path.sep.join(["output", "warmup.png"])