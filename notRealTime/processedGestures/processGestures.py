import numpy as np
import matplotlib.pyplot as plt
import csv
import sklearn
import pandas as pd
import glob

filePath = "data"
allFiles = glob.glob(filePath + "/*.csv")

## Store Tuples to array
# Collect data
allData = []*len(allFiles)
k = 0
for i in allFiles:
    identifiers = []
    with open(i, 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                row = map(float, row)
                row[0] = int(row[0])
                identifiers.append(tuple(row))
    allData.append(identifiers)
    k = k + 1

# Classify gestures (swipe and two taps)
counter = []
counterArray = [[]]*len(allData)
k = 0
blockCount = 0
count = 0
for i in allData:
    for j in range(0, len(i)):
        index = i[j][1]
        id_ = i[j][0]
        previous_index = i[j-1][1]
        previous_id = i[j-1][0]
        if abs(index - previous_index == 1):
            count = count + 1
        elif count > 0:
            counter.append(tuple((previous_id, count, previous_index)))
            count = 0

    counterArray[k] = counter
    counter = []
    k = k + 1
    count = 0

# Detect Gestures
        # Detect swipe
        # Detect two taps
        # Detect swirl

#    twotaps = np.array([])
#    # Detect swipe
#    for i in range(1, len(swipe_count)):
#        if abs(swipe_count[i] - swipe_count[i-1] < 3 and swipe_count[i] - swipe_count[i-1] > 0):
#            count += 1
#        elif abs(swipe_count[i] - swipe_count[i-1] > 10 and count > 1):
#            print 'swipe'
#            twotaps = np.append(twotaps, swipe_count[i] - swipe_count[i-1])
#            swipe += 1
#            count = 0
#         # Detect blocks
#         #if (swipe % 2 == 0):

# Cluster plot
x_swipe, y_swipe = [], []
for i in allData[3]:
    x_swipe.append(i[1])
    y_swipe.append(i[0])

plt.scatter(x_swipe, y_swipe, marker = 'o')

x_taps, y_taps = [], []
for i in allData[0]:
    x_taps.append(i[1])
    y_taps.append(i[0])
plt.scatter(x_taps, y_taps, marker = '*')

x_block, y_block = [], []
for i in allData[1]:
    x_block.append(i[1])
    y_block.append(i[0])

plt.scatter(x_block, y_block, marker = 'x')

x_swirl, y_swirl = [], []
for i in allData[2]:
    x_swirl.append(i[1])
    y_swirl.append(i[0])
plt.scatter(x_swirl, y_swirl, marker = '+')

plt.show()
