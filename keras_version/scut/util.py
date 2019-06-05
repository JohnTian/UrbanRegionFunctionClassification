# -*- encoding:utf-8 -*-
import cv2
import random
import datetime
import numpy as np
import pandas as pd

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

def visit2arrayAndNormal(filePath):
    table = pd.read_csv(filePath, header=None, sep='\t')
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
    return init / np.max(init)


def randomCropAndNormal(image_path, h=88, w=88):
    im = cv2.imread(image_path)
    height, width, _ = im.shape
    y = random.randint(1, height - h)
    x = random.randint(1, width - w)
    crop = im[y:y+h, x:x+w]
    return crop / 255.0
