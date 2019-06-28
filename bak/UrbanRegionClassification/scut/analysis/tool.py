# -*- encoding:utf-8 -*-
import os
import cv2
import sys
import codecs
import numpy as np
import json
import matplotlib.pyplot as plt

class RemoteSenseImage(object):
    
    def __init__(self, path, topk):
        self.path = path
        self.topk = topk
    
    def ComputeTheTopKSalientRegions(self):
        """
        Compute the top K salient regions of file.
        """
        image = cv2.imread(self.path)
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        _, saliencyMap = saliency.computeSaliency(image)
        # print(np.size(saliencyMap))
        # print(saliencyMap)
        saliencyMap = (saliencyMap * 255).astype("uint8")
        self.smap = saliencyMap
        cv2.imwrite(self.path[:-4]+"_saliencyMap.png", saliencyMap)

    def ShowSaliencyMap(self, t=0):
        cv2.imshow("saliency map", self.smap)
        cv2.waitKey(t*1000)

class Visit(object):

    def __init__(self, path, topk):
        self.path = path
        self.topk = topk
        self.data = dict()

    def drawImage(self):
        """
        Draw image by dataDict before topK
        """
        topK = self.topk
        dataDict = self.data
        # Sorted by value
        sortedDict = sorted(dataDict.items(), key=lambda item:item[0])
        time = [t[0] for t in sortedDict[-topK:]]
        y = [t[1] for t in sortedDict[-topK:]]

        # Set figure
        plt.figure(figsize=(16, 9))
        plt.title("Visits vs Times of 031934_001")
        plt.xlabel("Times")
        plt.ylabel("Visits")
        
        # Rotate the x index
        # x = range(len(time))
        # plt.xticks(x, time)
        # _, labels = plt.xticks()
        # plt.setp(labels, rotation=45)

        # Draw image
        # plt.plot(time, y, label="visits", linewidth=1, color='blue', marker='o', markerfacecolor='red', markersize=10)
        plt.bar(time, y, width=0.3, bottom=None, label="visits", align='center')

        # Put value in image
        for a, b in zip(time, y):
            plt.text(a, b, b, ha='center', va='bottom')

        # Show legend
        plt.legend()
        plt.show()

    def getDayVisits(self):
        """
        {
            "yearmonthday": numOfTimes
        }
        """
        pathOfTxt = self.path
        with codecs.open(pathOfTxt, 'r', 'utf-8') as fi:
            rootDict = dict()
            for line in fi.readlines():
                # 4823692475484e0f	20181216&19|20|21|23,20181217&09|10|12
                _, visitTimes = line.split('\t')
                for vt in visitTimes.split(','):
                    # yearMonthDay, times = vt.split('&')
                    yearMonthDay, _ = vt.split('&')
                    preVal = rootDict.get(yearMonthDay, 0)
                    # rootDict[yearMonthDay] = preVal + len(times.split('|'))
                    rootDict[yearMonthDay] = preVal + 1
            self.data = rootDict
    
    def getHourVisits(self):
        """
        {
            "0-24": numOfTimes
        }
        """
        pathOfTxt = self.path
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
            self.data = rootDict 
    
    def getDetailVisits(self):
        """
        {
            "year": 
            {
                "month":
                {
                    "day":
                    {
                        "0": numOfUserIDs0,
                        "1": numOfUserIDs1,
                        ......
                        "24": numOfUserIDs24
                    }
                }
            }
        }
        """
        pathOfTxt = self.path
        with codecs.open(pathOfTxt, 'r', 'utf-8') as fi:
            rootDict = dict()
            for line in fi.readlines():
                # 4823692475484e0f	20181216&19|20|21|23,20181217&09|10|12
                _, visitTimes = line.split('\t')
                for vt in visitTimes.split(','):
                    yearMonthDay, times = vt.split('&')
                    year, month, day = yearMonthDay[:4], yearMonthDay[4:6], yearMonthDay[6:]
                    rootDict[year]   = rootDict.get(year, {})
                    rootDict[year][month] = rootDict[year].get(month, {})
                    # day update
                    rootDict[year][month][day] = rootDict[year][month].get(day, {})
                    dayDict = rootDict[year][month][day]
                    for t in times.split('|'):
                        time = t.strip('\n')
                        preVal = dayDict.get(time, 0)
                        dayDict[time] =  preVal + 1
            self.data = rootDict

    def getYMDHVisits(self):
        """
        {
            "yearmonthday": 
            {
                "0": numOfUserIDs0,
                "1": numOfUserIDs1,
                ......
                "24": numOfUserIDs24
            }
        }
        """
        pathOfTxt = self.path
        with codecs.open(pathOfTxt, 'r', 'utf-8') as fi:
            rootDict = dict()
            for line in fi.readlines():
                # 4823692475484e0f	20181216&19|20|21|23,20181217&09|10|12
                _, visitTimes = line.split('\t')
                for vt in visitTimes.split(','):
                    yearMonthDay, times    = vt.split('&')
                    rootDict[yearMonthDay] = rootDict.get(yearMonthDay, {})
                    # day update
                    for t in times.split('|'):
                        time = t.strip('\n')
                        preVal = rootDict[yearMonthDay].get(time, 0)
                        rootDict[yearMonthDay][time] =  preVal + 1
            self.data = rootDict


if __name__ == '__main__':

    ## Visits
    pathOfTxt = "031934_001.txt"
    visit = Visit(pathOfTxt, 25)
    visit.getYMDHVisits()
    # visit.drawImage()
    # print(json.dumps(visit.data))
    data = visit.data
    y = []
    x = range(24)
    for k, v in data.items():
        y.append(k)
    ## Remote sense
    # pathOfImg = "031934_001.jpg"
    # remoteSenseImage = RemoteSenseImage(pathOfImg, 25)
    # remoteSenseImage.ComputeTheTopKSalientRegions()
    # remoteSenseImage.ShowSaliencyMap()