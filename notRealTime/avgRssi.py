import numpy as np
import csv
from matplotlib import pyplot as plt
import glob
import pandas as pd

filePath = "rssiData/rsbits"
allFiles = glob.glob(filePath + "/*noiseFloor.csv")

# Read all files and create dataset
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
rssiData = pd.concat(list_)

# r1 = receiver 1, r2 = receiver 2, etc ...
rssiData.columns = ['r1']

# Select RSSI values from each receiver
r1 = np.asarray(rssiData["r1"])

filter_bw = 20
sample_period = 25e-6
bitDuration = 0.01 # seconds
timeFrame = pd.DataFrame()
bitFrame = [[]]
for column in rssiData:
    rssiVals = rssiData[column]

    # Remove bad samples
    rssi = []
    dropCount = 0
    for i in range(0, len(rssiVals)):
        if rssiVals[i] < -80 and rssiVals[i] > -130:
            rssi.append(rssiVals[i])
        else:
            dropCount += 1
    rssiVals = rssi

meanRssi = np.mean(rssiVals)
