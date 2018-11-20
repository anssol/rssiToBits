import numpy as np
import sys
import select
import os
import pandas as pd
from matplotlib import pyplot as plt

# Normalizes data to between 0 and 1
def normalize(x):
    return (x - min(x))/(max(x) - min(x))

def movingAverage(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N 

# Define Sample Period and Frequency of CC1310
sample_period = 250e-6

# Open FIFO file
fifo = open("rssivals", "rb")

# Define parameters
#bufferlen = 2000 # The buffer length in CC1310 code
bufferlen = 8500 # The buffer length in CC1310 code
tresh = 0.2 
N = 1 # Window size of filter

rssi = np.array([])
raw_rssi = np.array([])
plt.ion()

while True:
    select.select([fifo],[],[fifo]) # Waits for writer
    line = fifo.readline()
    line = line.strip()
    if line != '':
        try:
            rssi = np.append(rssi, float(line))
        except ValueError:
            continue
    if len(rssi) == int(bufferlen):
        # Apply filter to signal
        rssi = movingAverage(rssi, N)
        rssivals = []
        for i in range(0, len(rssi)):
            if rssi[i] < -80 and rssi[i] > -130:
                rssivals.append(rssi[i])

        # Normalize data
        rssi = rssivals
        rssi = normalize(rssi)
        raw_rssi = rssi

        x = np.arange(0, len(rssi))*sample_period/10

        # Draw plot 
        plt.subplot(2, 1, 1)
        plt.plot(x, raw_rssi)
        plt.xticks([])
        plt.ylim([-0.2, 1.2])
        plt.yticks([0,1])
        plt.ylabel("Normalized Signal")
        
        # Replace data that are close to the noise floor
        # Todo: Try to make this more generic
        rssi[rssi < np.mean(rssi) + 0.2] = 0
        rssi[rssi > np.mean(rssi) + 0.2] = 1

        count = 0
        arr_i = []
        #for i in range(1,len(rssi)):
        #    if (rssi[i] == 1):
        #        count += 1
        #        #arr_i.append[i]
        #    if (rssi[i] == 0 and count > 0 and count*sample_period < 1e-3):
        #        #arr_i = []
        #        print count*sample_period
        
        plt.subplot(2, 1, 2)
        plt.plot(x, rssi)
        plt.xlabel('Time [s]', fontsize = 14)
        plt.ylim([-0.2, 1.2])
        plt.yticks([0,1])
        plt.ylabel("Digitized Signal")
        plt.draw()

        plt.pause(0.0001)
        plt.clf()

        # Prepare for next iteration
        rssi = np.array([])

        # Close FIFO file
        #fifo.close()
