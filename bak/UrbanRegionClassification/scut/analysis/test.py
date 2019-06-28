# -*- encoding:utf-8 -*-
import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from util import standardization
from util import showImage, getHourVisits, computeSalientRegions, drawImage


if __name__ == "__main__":
    # visit
    pathOfTxt = "031934_001.txt"
    visitsByHourTmp = getHourVisits(pathOfTxt)
    visitsByHour = dict(sorted(visitsByHourTmp.items(), key=lambda t: t[0]))
    # Sorted by value
    sortedDict = sorted(visitsByHour.items(), key=lambda item:item[0])
    hourX = [int(t[0]) for t in sortedDict]
    visitsY = [int(t[1]) for t in sortedDict]
    visitsY = standardization(visitsY)
    drawImage(hourX, visitsY, title="visits VS hours", xlabel="Hours", ylabel="Visits", dlabel="visit volume")
    
    # remote sense
    pathOfImg = "031934_001.jpg"
    salientMap = computeSalientRegions(pathOfImg)
    # salientMap = cv2.imread(pathOfImg, 0)
    salientMapInRow = np.reshape(salientMap, (1, 100*100))
    salient = salientMapInRow.tolist()[0]
    step = int(10000 / 24)
    rsY = [np.mean(salient[i:i+step]) for i in range(0, 10000, step)]
    rsY = standardization(rsY[:-1])
    rsX = range(len(rsY))
    drawImage(rsX, rsY, title="salient VS simulated time", xlabel="Simulated time", ylabel="Salient", dlabel="salient value")

    # Merge
    if len(hourX) == len(rsX):
        x = range(len(rsX))
        y = [0.7*visitsY[i] + 0.3*rsY[i] for i in range(len(x))]
        drawImage(x, y, title="visitSalient VS simulated time", xlabel="Simulated time", ylabel="visitSalient", dlabel="visit salient")