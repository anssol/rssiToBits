from __future__ import generators
import numpy as np
import csv
from matplotlib import pyplot as plt
import glob
import pandas as pd
from avgRssi import meanRssi

def generatePreambleSequence(preamble, samplesPerBit):
    preambleSequence = np.array([])
    for i in range(0, len(preamble)):
        preambleSequence = np.append(preambleSequence, [preamble[i]] * int(samplesPerBit))
    return preambleSequence

def KnuthMorrisPratt(text, pattern):

    '''Yields all starting positions of copies of the pattern in the text.
Calling conventions are similar to string.find, but its arguments can be
lists or iterators, not just strings, it returns all matches, not just
the first one, and it does not need the whole text in memory at once.
Whenever it yields, it will have read the text exactly up to and including
the match that caused the yield.'''

    # allow indexing into pattern and protect against change during yield
    pattern = list(pattern)

    # build table of shift amounts
    shifts = [1] * (len(pattern) + 1)
    shift = 1
    for pos in range(len(pattern)):
        while shift <= pos and pattern[pos] != pattern[pos-shift]:
            shift += shifts[pos-shift]
        shifts[pos+1] = shift

    # do the actual search
    startPos = 0
    matchLen = 0
    for c in text:
        while matchLen == len(pattern) or \
              matchLen >= 0 and pattern[matchLen] != c:
            startPos += shifts[matchLen]
            matchLen -= shifts[matchLen]
        matchLen += 1
        if matchLen == len(pattern):
            yield startPos


# Normalization
def normalize(x):
    return (x - min(x))/float((max(x) - min(x)))

# Filtering
def movingAverage(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

# Choose to include or not to include the preamble sequence
def extractBits(sequences, preambleSequence):
    stream = [[]]*len(sequences)
    k = 0
    for i in sequences:
        stream[k] = rssiVals[i:i+len(preambleSequence)+40]
        k += 1
    return stream

filePath = "rssiData/rsbits"
allFiles = glob.glob(filePath + "/*rssi_1.csv")

# Read all files and create dataset
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
rssiData = pd.concat(list_)

meanRssi = -80 # Hardcoded

# r1 = receiver 1, r2 = receiver 2, etc ...
rssiData.columns = ['r1']

# Select RSSI values from each receiver
r1 = np.asarray(rssiData["r1"])

# Process Data
filter_bw = 20
sample_period = 1e-3
bitDuration = 0.01 # seconds

# Generate Preamble sequence
samplesPerBit = bitDuration/sample_period
preamble = np.array([0, 1, 1, 0])
preambleSequence = generatePreambleSequence(preamble, samplesPerBit)

timeFrame = pd.DataFrame()
for column in rssiData:
    rssiVals = rssiData[column]

    # Remove bad samples
    rssi = []
    dropCount = 0
    for i in range(0, len(rssiVals)):
        if rssiVals[i] < -50 and rssiVals[i] > -130:
            rssi.append(rssiVals[i])
        else:
            dropCount += 1
    rssiVals = rssi
    
 #   # Filter
 #   rssiVals = movingAverage(rssiVals, filter_bw) 

    # Normalize 
#    rssiVals = normalize(rssiVals)

    #rssiVals = np.append(rssiVals, np.zeros(dropCount)) # Fix length
    rssiVals = np.append(rssiVals, [-120]*dropCount) # Fix length

    # Convert RSSI to binary (change later if needed)
    rssiVals = np.asarray(rssiVals)
    highs = rssiVals > meanRssi 
    lows = ~highs
    rssiVals[highs] = 1
    rssiVals[lows] = 0
    rssiData[column] = rssiVals

    # Generate Time sequence
    x = np.arange(0, len(rssiVals))*sample_period

    # Create dataFrame of time samples
    timeFrame[column+'_time'] = x

    # Find transitions between 0s and 1s 
    transitions = np.where(rssiVals[:-1] != rssiVals[1:])[0]
    transitionTimes = x[transitions]
    transitionVals = rssiVals[transitions]

    # Extract time domain information of events
    digital_events = np.round(transitionTimes[1:] - transitionTimes[:-1], 2)

    # Convert to bits
#    bitStream = []
#    for i in range(0, len(digital_events)):
#        if (i % 2 == 0):
#            if (digital_events[i] == bitDuration):
#                bitStream.extend([1])
#            elif (digital_events[i] == 2*bitDuration):
#                bitStream.extend([1,1])
#            elif (digital_events[i] == 3*bitDuration):
#                bitStream.extend([1,1,1])
#            elif (digital_events[i] == 4*bitDuration):
#                bitStream.extend([1,1,1,1])
#        else:
#            if (digital_events[i] == bitDuration):
#                bitStream.extend([0])
#            elif (digital_events[i] == 2*bitDuration):
#                bitStream.extend([0,0])
#            elif (digital_events[i] == 3*bitDuration):
#                bitStream.extend([0,0,0])
#            elif (digital_events[i] == 4*bitDuration):
#                bitStream.extend([0,0,0,0])
#            elif (digital_events[i] == 5*bitDuration):
#                bitStream.extend([0,0,0,0,0])
#            elif (digital_events[i] == 6*bitDuration):
#                bitStream.extend([0,0,0,0,0,0])
#            elif (digital_events[i] == 8*bitDuration):
#                bitStream.extend([0,0,0,0,0,0,0,0])
#            elif (digital_events[i] == 9*bitDuration): # due to rounding error
#                bitStream.extend([0,0,0,0,0,0,0,0,0])
#
#        # Store the bits
#        bitFrame.append(bitStream)

# Extract preamble sequence from received RSSI
sequences = []
for s in KnuthMorrisPratt(rssiVals, preambleSequence): sequences.append(s)

# Extract bits
# Todo: accommodate for indexes that are right next to each other;
# i.e. if the preamble is equal to the following bits
bitStream = extractBits(sequences, preambleSequence)

# Plot data
#plt.step(timeFrame['r1_time'][0:20000], rssiData['r1'][0:20000])
#plt.step(timeFrame['r1_time'][0:5000], rssiData['r1'][0:5000])
#plt.ylim([-0.2, 1.2])
#plt.ylabel("Digitized Signal")
#plt.show()
