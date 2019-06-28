# -*- encoding:utf-8 -*-
import cv2
import codecs
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


def showImage(winame, im, waitime):
    cv2.namedWindow(winame)
    cv2.imshow(winame, im)
    cv2.waitKey(1000*waitime)
    cv2.destroyWindow(winame)


def computeSalientRegions(path):
    """
    Compute the top K salient regions of file.
    """
    image = cv2.imread(path)
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    _, saliencyMap = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    return saliencyMap


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def drawImage(x, y, title, xlabel, ylabel="visits", dlabel="visit"):
    # Set figure
    # plt.figure(figsize=(16, 9))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(len(x)))
    
    # Rotate the x index
    # xt = range(len(x))
    # plt.xticks(xt, x)
    # _, labels = plt.xticks()
    # plt.setp(labels, rotation=45)

    # Draw image
    plt.plot(x, y, label=dlabel, linewidth=1, color='blue', marker='o', markerfacecolor='red', markersize=10)
    # plt.bar(x, y, width=0.3, bottom=None, label="visits", align='center')

    # Put value in image
    for a, b in zip(x, y):
        plt.text(a, b, round(b,2), ha='center', va='bottom')

    # Show legend
    plt.legend()
    plt.show()


def computeWhichDayInWeek(yearMonthDay):
    """
    已知年月日, 计算对应一周的星期几.
    20181001, 星期一
    """
    year, month, day = yearMonthDay[:4], yearMonthDay[4:6], yearMonthDay[6:]
    date = datetime.date(day=int(day), month=int(month), year=int(year))
    return date.weekday()

def getWeekVisits(pathOfTxt):
    """
    {
        "week": numOfTimes
    }
    """
    with codecs.open(pathOfTxt, 'r', 'utf-8') as fi:
        rootDict = dict()
        for line in fi.readlines():
            # 4823692475484e0f	20181216&19|20|21|23,20181217&09|10|12
            _, visitTimes = line.split('\t')
            for vt in visitTimes.split(','):
                yearMonthDay, _ = vt.split('&')
                whichDayInWeek = computeWhichDayInWeek(yearMonthDay)
                preVal = rootDict.get(whichDayInWeek, 0)
                rootDict[whichDayInWeek] = preVal + 1
        return rootDict

def getMonthVisits(pathOfTxt):
    """
    {
        "month": numOfTimes
    }
    """
    with codecs.open(pathOfTxt, 'r', 'utf-8') as fi:
        rootDict = dict()
        for line in fi.readlines():
            # 4823692475484e0f	20181216&19|20|21|23,20181217&09|10|12
            _, visitTimes = line.split('\t')
            for vt in visitTimes.split(','):
                yearMonthDay, _ = vt.split('&')
                year, month, day = yearMonthDay[:4], yearMonthDay[4:6], yearMonthDay[6:]
                preVal = rootDict.get(month, 0)
                rootDict[month] = preVal + 1
        return rootDict


def getDayVisits(pathOfTxt):
    """
    {
        "day": numOfTimes
    }
    """
    with codecs.open(pathOfTxt, 'r', 'utf-8') as fi:
        rootDict = dict()
        for line in fi.readlines():
            # 4823692475484e0f	20181216&19|20|21|23,20181217&09|10|12
            _, visitTimes = line.split('\t')
            for vt in visitTimes.split(','):
                yearMonthDay, _ = vt.split('&')
                year, month, day = yearMonthDay[:4], yearMonthDay[4:6], yearMonthDay[6:]
                preVal = rootDict.get(day, 0)
                rootDict[day] = preVal + 1
        return rootDict


def getHourVisits(pathOfTxt):
    """
    {
        "hour": numOfTimes
    }
    """
    with codecs.open(pathOfTxt, 'r', 'utf-8') as fi:
        rootDict = dict()
        for line in fi.readlines():
            # 4823692475484e0f	20181216&19|20|21|23,20181217&09|10|12
            _, visitTimes = line.split('\t')
            for vt in visitTimes.split(','):
                _, times = vt.split('&')
                for t in times.split('|'):
                    time = t.strip('\n')
                    preVal = rootDict.get(time, 0)
                    rootDict[time] = preVal + 1
        return rootDict


def setCSVdata(pathOfTxt, fmethod, label, csvRow):
    visitByTime = fmethod(pathOfTxt)
    ddict = dict(sorted(visitByTime.items(), key=lambda item:item[0]))
    for k, v in ddict.items():
        k = str(int(k)) + label
        csvRow[k] = v


def computeOneRowForCSV(pathOfTxt):
    # 设置字典默认值
    keys = [
        'id', 'area',
        '0h', '1h', '2h', '3h', '4h', '5h', '6h', '7h', 
        '8h', '9h', '10h', '11h', '12h', '13h', '14h', '15h', 
        '16h', '17h', '18h', '19h', '20h', '21h', '22h', '23h',
        '1d', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', '10d',
        '11d', '12d', '13d', '14d', '15d', '16d', '17d', '18d', '19d', '20d',
        '21d', '22d', '23d', '24d', '25d', '26d', '27d', '28d', '29d', '30d', '31d',
        '0w', '1w', '2w', '3w', '4w', '5w', '6w',
        '1m', '2m', '3m', '10m', '11m', '12m'
    ]
    csvRow = OrderedDict()
    for k in keys:
        csvRow[k] = 'null'

    idx1 = pathOfTxt.find('_')
    idx2 = pathOfTxt.find('.')
    area = pathOfTxt[idx1+1:idx2]
    csvRow['id'] = pathOfTxt[:-4]
    csvRow['area'] = int(area) - 1 # for one-hot encoding 0-8 => 1-9

    setCSVdata(pathOfTxt, getHourVisits, 'h', csvRow)
    setCSVdata(pathOfTxt, getDayVisits, 'd', csvRow)
    setCSVdata(pathOfTxt, getWeekVisits, 'w', csvRow)
    setCSVdata(pathOfTxt, getMonthVisits, 'm', csvRow)

    return list(csvRow.values())