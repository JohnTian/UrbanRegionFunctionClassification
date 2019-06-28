# -*- encoding:utf-8 -*-
import os
import sys
import json
from write2excel import dowrite
from util import getHourVisits, getDayVisits, getMonthVisits, drawImage
from util import getWeekVisits

categoryID2Areas = {
    "001": "Residential area",
    "002": "School",
    "003": "Industrial park",
    "004": "Railway station",
    "005": "Airport",
    "006": "Park",
    "007": "Shopping area",
    "008": "Administrative district",
    "009": "Hospital"
}

def addDict(a, b):
    """
    return dict for a and b. do add for same key, do append for different key.
    """
    r = dict()
    r.update(a)
    for k, v in b.items():
        if k in r.keys():
            r[k] = r[k] + v
        else:
            r[k] = v
    return r

def getVisits(argvs, filterMethod, jsonFile):
    """
    Get visits from argvs folder, and filter by filterMethod.
    Return Dict object.
    """
    areas = sorted(os.listdir(argvs))
    areasDict = dict()
    for area in areas:
        areaVisitDict = {}
        areaPath = os.path.join(argvs, area)
        for areaFile in os.listdir(areaPath):
            if ".txt" not in areaFile:
                continue
            areaFilePath = os.path.join(areaPath, areaFile)
            tmpDict = filterMethod(areaFilePath)
            areaVisitDict = addDict(areaVisitDict, tmpDict)
        areasDict[area] = areaVisitDict
        print(areaPath + "-->" + "done!")
    # save all visits
    with open(jsonFile, "w") as f:
        json.dump(areasDict, f)   


def doWrite2Excel(jsonFile, titles, sheename, excelFile):
    """
    Do write data from json file to excel file.
    """
    with open(jsonFile, "r") as fi:
        data = json.load(fi)
        lines = []
        for area, v in data.items():
            line = []
            line.append(categoryID2Areas[area])
            sortedV = sorted(v.items(), key=lambda item:item[0])
            for v in sortedV:
                line.append(v[1])
            lines.append(line)
        dowrite(titles, sheename, lines, excelFile)


if __name__ == "__main__":
    argvs = "train"
    ## Hours
    # getVisits(argvs, getHourVisits, "HourVisits.json")
    titlesHour = ["Area"]
    for i in range(24):
        titlesHour.append(str(i).zfill(2))
    doWrite2Excel("HourVisits.json", titlesHour, "visitByHour", "visits.xls")

    ## Days
    # getVisits(argvs, getDayVisits, "DayVisits.json")
    titlesDay = ["Area"]
    for i in range(1,32):
        titlesDay.append(str(i).zfill(2))
    doWrite2Excel("DayVisits.json", titlesDay, "visitByDay", "visits.xls")

    ## Weeks
    # getVisits(argvs, getWeekVisits, "WeekVisits.json")
    whichDayInWeek = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday"
    }
    titlesWeek = ["Area"]
    for i in range(7):
        titlesWeek.append(whichDayInWeek[i])
    doWrite2Excel("WeekVisits.json", titlesWeek, "visitByWeek", "visits.xls")

    ## Months
    # getVisits(argvs, getMonthVisits, "MonthVisits.json")
    titlesMonth = ["Area", "01", "02", "03", "10", "11", "12"]
    doWrite2Excel("MonthVisits.json", titlesMonth, "visitByMonth", "visits.xls")

