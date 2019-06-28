# -*- encoding:utf-8 -*-
import os
import sys
import cv2
import time
import random
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf


# Refer repo: https://github.com/czczup/UrbanRegionFunctionClassification.git
# "00": 0, "01": 1, ..., "23": 23
str2int = {}
for i in range(24):
	str2int[str(i).zfill(2)] = i

# 访问记录内的时间从2018年10月1日起，共182天即26周, 每周7天, 将日期按日历排列
date2position = {}
datestr2dateint = {}
for i in range(182):
	date = datetime.date(day=1, month=10, year=2018)+datetime.timedelta(days=i)
	date_int = int(date.__str__().replace("-", ""))
	# 20181001: [0, 0]
	date2position[date_int] = [i%7, i//7]
	# "20181001": 20181001
	datestr2dateint[str(date_int)] = date_int

def visit2array(table):
    strings = table[1]
    # 7天, 26周, 每天24小时
    init = np.zeros((7, 26, 24))
    for string in strings:
        temp = []
        for item in string.split(','):
            temp.append([item[0:8], item[9:].split("|")])
        for date, visit_lst in temp:
            # 第y周的第x天的到访总人数为value
            x, y = date2position[datestr2dateint[date]]
            for visit in visit_lst:
                init[x][y][str2int[visit]] += 1
    return init


def creatEval(dataFolder):
    data = [f for f in os.listdir(dataFolder) if f[-3:] == 'txt']
    data = sorted(data, key=lambda d:int(d.split('.')[0]))
    evalVisit = [os.path.join(dataFolder, f) for f in data]
    evalImage = [f.replace('txt', 'jpg') for f in evalVisit]
    return {'visit':evalVisit}, {'image':evalImage}


def balanceData(dataFolder, numOfCls=10, ratio=0.25):
    data = [f for f in os.listdir(dataFolder) if f[-3:] == 'txt']
    data = sorted(data, key=lambda d:int(d.split('_')[-1].split('.')[0]))
    # 记录各个类中文件地址
    files = {}
    for areaID in range(1, numOfCls):
        files[areaID] = []
    for f in data:
        fPath = os.path.join(dataFolder, f)
        areaID = int(fPath.split('_')[-1].split('.')[0])
        files[areaID].append(fPath)
    # 将files拆分为trainData和validData
    trainData = {}
    validData = {}
    minNum = min([len(v) for _, v in files.items()])
    numOfAvgValid = int(minNum*ratio)
    for areaID in range(1, numOfCls):
        validData[areaID] = random.sample(files[areaID], numOfAvgValid)
        trainData[areaID] = list(set(files[areaID]) - set(validData[areaID]))
    # balance number of class for trainData
    trainBalanceData = {}
    trainBalanceData.update(trainData)
    maxNum = max([len(v) for _, v in trainData.items()])
    for i in range(1, numOfCls):
        for _ in range(maxNum - len(trainData[i])):
            trainBalanceData[i].append(trainData[i][random.randint(0, len(trainData[i])-1)])
    # 后处理
    trainBalanceVisitData = trainBalanceData
    trainBalanceImageData = {}
    trainLabel = []
    for k, v in trainBalanceVisitData.items():
        trainBalanceImageData[k] = [t.replace('txt', 'jpg') for t in v]
        trainLabel.extend([k]*len(v))

    validVisitData = validData
    validImageData = {}
    validLabel = []
    for k, v in validVisitData.items():
        validImageData[k] = [t.replace('txt', 'jpg') for t in v]
        validLabel.extend([k]*len(v))

    return (trainBalanceVisitData, trainBalanceImageData, trainLabel), (validVisitData, validImageData, validLabel)


def genNumpyObjectForVisit(visitDict):
    start_time = time.time()
    visits = []
    for _, v in visitDict.items():
        for fPath in v:
            table = pd.read_table(fPath, header=None)
            visit = visit2array(table)
            # 7*26*24 -> 24*26*7
            #tmp = visit.transpose(2, 1, 0)
            # 24*26*7 -> 32*32*7
            #elm = np.pad(tmp, ((4,4), (3,3), (0,0)), mode='constant', constant_values=0)
            #visits.append(elm)
            visits.append(visit)
    print("using time:%.2fs"%(time.time()-start_time))
    return np.array(visits)

def genNumpyObjectForImage(imageDict):
    start_time = time.time()
    images = []
    for _, v in imageDict.items():
        for imPath in v:
            image  = cv2.imread(imPath)
            images.append(image)
    print("using time:%.2fs"%(time.time()-start_time))
    return np.array(images)

# Helperfunctions to make your feature definition more readable
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write2tfrecord(visits, images, labels, tfrecordPath):
    # create filewriter
    writer = tf.python_io.TFRecordWriter(tfrecordPath)
    lenth = visits.shape[0]
    print('visits shape:', np.shape(visits))
    print('images shape:', np.shape(images))
    print('labels shape:', np.shape(labels))
    for i in range(lenth):
        visit = visits[i]
        image = images[i]
        label = labels[i]
        # Define the features of your tfrecord
        feature = {
            'visit':  _bytes_feature(tf.compat.as_bytes(visit.tostring())),
            'image':  _bytes_feature(tf.compat.as_bytes(image.tostring())),
            'label':  _int64_feature(label)
        }
        # Serialize to string and write to file
        example = tf.train.Example(
            features=tf.train.Features(feature=feature)
        )
        writer.write(example.SerializeToString())

def write2tfrecordForEval(visits, images, tfrecordPath):
    # create filewriter
    writer = tf.python_io.TFRecordWriter(tfrecordPath)
    lenth = visits.shape[0]
    print('eval visits shape:', np.shape(visits))
    print('eval images shape:', np.shape(images))
    for i in range(lenth):
        visit = visits[i]
        image = images[i]
        # Define the features of your tfrecord
        feature = {
            'visit':  _bytes_feature(tf.compat.as_bytes(visit.tostring())),
            'image':  _bytes_feature(tf.compat.as_bytes(image.tostring()))
        }
        # Serialize to string and write to file
        example = tf.train.Example(
            features=tf.train.Features(feature=feature)
        )
        writer.write(example.SerializeToString())


def genTFrecord(dataFolder, evalFolder, npyFolder, tfrecordFolder):
    if not os.path.exists(dataFolder):
        print(dataFolder, "not exist!")
    if not os.path.exists(evalFolder):
        print(evalFolder, "not exist!")
    if not os.path.exists(npyFolder):
        os.makedirs(npyFolder)
    if not os.path.exists(tfrecordFolder):
        os.makedirs(tfrecordFolder)

    (trainBalanceVisitData, trainBalanceImageData, trainLabel), (validVisitData, validImageData, validLabel) = balanceData(dataFolder)

    validVisit = genNumpyObjectForVisit(validVisitData)
    np.save(os.path.join(npyFolder, "validVisit.npy"), validVisit)
    validImage = genNumpyObjectForImage(validImageData)
    np.save(os.path.join(npyFolder, "validImage.npy"), validImage)
    validLabel = np.array(validLabel)
    np.save(os.path.join(npyFolder, "validLabel.npy"), validLabel)
    write2tfrecord(validVisit, validImage, validLabel, os.path.join(tfrecordFolder, "valid.tfrecord"))

    evalVisit, evalImage = creatEval(evalFolder)
    evalVisit = genNumpyObjectForVisit(evalVisit)
    evalImage = genNumpyObjectForImage(evalImage)
    np.save(os.path.join(npyFolder, "evalVisit.npy"), evalVisit)
    np.save(os.path.join(npyFolder, "evalImage.npy"), evalImage)
    write2tfrecordForEval(evalVisit, evalImage, os.path.join(tfrecordFolder, "eval.tfrecord"))

    trainVisit = genNumpyObjectForVisit(trainBalanceVisitData)
    np.save(os.path.join(npyFolder, "trainVisit.npy"), trainVisit)
    trainImage = genNumpyObjectForImage(trainBalanceImageData)
    np.save(os.path.join(npyFolder, "trainImage.npy"), trainImage)
    trainLabel = np.array(trainLabel)
    np.save(os.path.join(npyFolder, "trainLabel.npy"), trainLabel)
    write2tfrecord(trainVisit, trainImage, trainLabel, os.path.join(tfrecordFolder, "train.tfrecord"))


if __name__ == '__main__':
    dataFolder = "../data/train/"
    evalFolder = "../data/eval/"
    npyFolder = "../data/npy/"
    tfrecordFolder = "../data/tfrecord/"
    genTFrecord(dataFolder, evalFolder, npyFolder, tfrecordFolder)
