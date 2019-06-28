# -*- encoding:utf-8 -*-
import os
import cv2
import csv
import numpy as np
from .util import computeOneRowForCSV


if __name__ == "__main__":
    
    dataFolder = 'test'
    with open('test.csv', 'w') as f:
        csvWriter = csv.writer(f)
        # title = [
        #     'id', 'area',
        #     '0h', '1h', '2h', '3h', '4h', '5h', '6h', '7h', 
        #     '8h', '9h', '10h', '11h', '12h', '13h', '14h', '15h', 
        #     '16h', '17h', '18h', '19h', '20h', '21h', '22h', '23h',
        #     '1d', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', '10d',
        #     '11d', '12d', '13d', '14d', '15d', '16d', '17d', '18d', '19d', '20d',
        #     '21d', '22d', '23d', '24d', '25d', '26d', '27d', '28d', '29d', '30d', '31d',
        #     '0w', '1w', '2w', '3w', '4w', '5w', '6w',
        #     '1m', '2m', '3m', '10m', '11m', '12m'
        # ]
        # csvWriter.writerow(title)
        data = [f for f in os.listdir(dataFolder) if f[-3:] == 'txt']
        data = sorted(data, key=lambda d:int(d.split('_')[-1].split('.')[0]))
        i = 0
        for f in data:
            fPath = os.path.join(dataFolder, f)
            csvRow = computeOneRowForCSV(fPath)
            csvWriter.writerow(csvRow)
            i += 1
            if i % 500 == 0:
                print(i, '-->', 'done!')