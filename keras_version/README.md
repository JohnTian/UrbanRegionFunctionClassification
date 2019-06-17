# Keras version for Urban region classification

- [Keras version for Urban region classification](#keras-version-for-urban-region-classification)
  - [0 Requirement](#0-requirement)
  - [1 Structure](#1-structure)
  - [2 Steps](#2-steps)
    - [2.1 Build dataset](#21-build-dataset)
    - [2.2 Training the model and test](#22-training-the-model-and-test)

## 0 Requirement

- Python Libs

```BASH
pip
Keras
opencv-python
tensorflow
h5py
imutils
pydot    # save model structure in picture
graphviz # save model structure in picture
matplotlib
numpy
pandas
Pillow
scikit-learn
scipy
```

- Libs

```BASH
sudo apt-get install graphviz
```

## 1 Structure

```BASH
➜  keras_version git:(master) tree -L 1
.
├── build_dataset.py
├── dataset           # to store datas after running `build_dataset.py`
├── log               # to store files for tensorboard after running `train.py`
├── saved_models      # to store model file
├── scut              # python module
├── train.py
└── UrbanRegionData   # Original data
```

In **UrbanRegionData** folder:

```BASH
➜  keras_version git:(master) tree UrbanRegionData --dirsfirst --filelimit 10 
UrbanRegionData
├── test_eval [20000 entries exceeds filelimit, not opening dir]
└── train
    ├── 001 [19084 entries exceeds filelimit, not opening dir]
    ├── 002 [15076 entries exceeds filelimit, not opening dir]
    ├── 003 [7180 entries exceeds filelimit, not opening dir]
    ├── 004 [2716 entries exceeds filelimit, not opening dir]
    ├── 005 [6928 entries exceeds filelimit, not opening dir]
    ├── 006 [11014 entries exceeds filelimit, not opening dir]
    ├── 007 [7034 entries exceeds filelimit, not opening dir]
    ├── 008 [5234 entries exceeds filelimit, not opening dir]
    └── 009 [5734 entries exceeds filelimit, not opening dir]
```

## 2 Steps

### 2.1 Build dataset

```BASH
python build_dataset.py
```

**Result**:

```BASH
➜  keras_version git:(master) tree dataset --dirsfirst --filelimit 10
dataset
├── image
│   ├── evaluation
│   │   ├── Administrative district [136 entries exceeds filelimit, not opening dir]
│   │   ├── Airport [136 entries exceeds filelimit, not opening dir]
│   │   ├── Hospital [136 entries exceeds filelimit, not opening dir]
│   │   ├── Industrial park [136 entries exceeds filelimit, not opening dir]
│   │   ├── Park [136 entries exceeds filelimit, not opening dir]
│   │   ├── Railway station [136 entries exceeds filelimit, not opening dir]
│   │   ├── Residential area [136 entries exceeds filelimit, not opening dir]
│   │   ├── School [136 entries exceeds filelimit, not opening dir]
│   │   └── Shopping area [136 entries exceeds filelimit, not opening dir]
│   ├── training
│   │   ├── Administrative district [2345 entries exceeds filelimit, not opening dir]
│   │   ├── Airport [3192 entries exceeds filelimit, not opening dir]
│   │   ├── Hospital [2595 entries exceeds filelimit, not opening dir]
│   │   ├── Industrial park [3318 entries exceeds filelimit, not opening dir]
│   │   ├── Park [5235 entries exceeds filelimit, not opening dir]
│   │   ├── Railway station [1086 entries exceeds filelimit, not opening dir]
│   │   ├── Residential area [9270 entries exceeds filelimit, not opening dir]
│   │   ├── School [7266 entries exceeds filelimit, not opening dir]
│   │   └── Shopping area [3245 entries exceeds filelimit, not opening dir]
│   └── validation
│       ├── Administrative district [136 entries exceeds filelimit, not opening dir]
│       ├── Airport [136 entries exceeds filelimit, not opening dir]
│       ├── Hospital [136 entries exceeds filelimit, not opening dir]
│       ├── Industrial park [136 entries exceeds filelimit, not opening dir]
│       ├── Park [136 entries exceeds filelimit, not opening dir]
│       ├── Railway station [136 entries exceeds filelimit, not opening dir]
│       ├── Residential area [136 entries exceeds filelimit, not opening dir]
│       ├── School [136 entries exceeds filelimit, not opening dir]
│       └── Shopping area [136 entries exceeds filelimit, not opening dir]
└── visit
    ├── evaluation
    │   ├── Administrative district [136 entries exceeds filelimit, not opening dir]
    │   ├── Airport [136 entries exceeds filelimit, not opening dir]
    │   ├── Hospital [136 entries exceeds filelimit, not opening dir]
    │   ├── Industrial park [136 entries exceeds filelimit, not opening dir]
    │   ├── Park [136 entries exceeds filelimit, not opening dir]
    │   ├── Railway station [136 entries exceeds filelimit, not opening dir]
    │   ├── Residential area [136 entries exceeds filelimit, not opening dir]
    │   ├── School [136 entries exceeds filelimit, not opening dir]
    │   └── Shopping area [136 entries exceeds filelimit, not opening dir]
    ├── training
    │   ├── Administrative district [2345 entries exceeds filelimit, not opening dir]
    │   ├── Airport [3192 entries exceeds filelimit, not opening dir]
    │   ├── Hospital [2595 entries exceeds filelimit, not opening dir]
    │   ├── Industrial park [3318 entries exceeds filelimit, not opening dir]
    │   ├── Park [5235 entries exceeds filelimit, not opening dir]
    │   ├── Railway station [1086 entries exceeds filelimit, not opening dir]
    │   ├── Residential area [9270 entries exceeds filelimit, not opening dir]
    │   ├── School [7266 entries exceeds filelimit, not opening dir]
    │   └── Shopping area [3245 entries exceeds filelimit, not opening dir]
    └── validation
        ├── Administrative district [136 entries exceeds filelimit, not opening dir]
        ├── Airport [136 entries exceeds filelimit, not opening dir]
        ├── Hospital [136 entries exceeds filelimit, not opening dir]
        ├── Industrial park [136 entries exceeds filelimit, not opening dir]
        ├── Park [136 entries exceeds filelimit, not opening dir]
        ├── Railway station [136 entries exceeds filelimit, not opening dir]
        ├── Residential area [136 entries exceeds filelimit, not opening dir]
        ├── School [136 entries exceeds filelimit, not opening dir]
        └── Shopping area [136 entries exceeds filelimit, not opening dir]

62 directories, 0 files
```

### 2.2 Training the model and test

```BASH
python train.py
```

**Result**:

```BASH
➜  keras_version git:(master) ls -hl log saved_models
log:
total 17M
-rw-rw-r-- 1 epbox epbox  14M 6月  10 12:03 events.out.tfevents.1560137533.epbox-All-Series
-rw-rw-r-- 1 epbox epbox 2.6M 6月  10 11:32 urbanRegion.png

saved_models:
total 81M
-rw-rw-r-- 1 epbox epbox 81M 6月  10 11:40 urbanRegion.h5
-rw-rw-r-- 1 epbox epbox 30K 6月  10 12:03 warmup.png
```
