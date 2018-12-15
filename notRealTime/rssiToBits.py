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

# With Preamble
def extractBits(sequences, preambleSequence, samplesPerBit, nbits):
    stream = [[]]*len(sequences)
    compressedStream = [[]]*len(sequences)
    bitStream = []
    k = 0
    try:
        for i in sequences:
            stream[k] = rssiVals[i:i+len(preambleSequence)+int(nbits*samplesPerBit)]
            # Compress the bits to nBit-format
            for j in range(0, 4+nbits):
                bit = stream[k][j + j*samplesPerBit]
                bitStream.append(bit)
            compressedStream[k] = bitStream
            bitStream = []
            k += 1
    except:
        pass
    return stream, compressedStream

# No preamble
def extractDataBits(sequences, preambleSequence, samplesPerBit, nbits):
    stream = [[]]*len(sequences)
    compressedStream = [[]]*len(sequences)
    bitStream = []
    k = 0
    try:
        for i in sequences:
            stream[k] = rssiVals[i+len(preambleSequence):i+len(preambleSequence)+int(nbits*samplesPerBit)-1]
            # Compress the bits to nBit-format
            for j in range(0, nbits):
                bit = stream[k][j + j*samplesPerBit]
                bitStream.append(bit)
            compressedStream[k] = bitStream
            bitStream = []
            k += 1
    except:
        pass
    return stream, compressedStream

## Move to another file later ##
def bool2int(x):
    y = 0
    for i,j in enumerate(x):
        y += j << i
    return y

def getIdentifiers(decimalArray):
    #idArray = np.array([], dtype='string')
    idArray = []
    #idArray = np.zeros(len(decimalArray))
    #block_count = np.array([])
    #swipe_count = []
    k = 0
    for i in decimalArray:
        if i == 1:
            #idArray = np.append(idArray, '1')

            idArray.append(tuple([1, k]))
        elif i == 2:
            #idArray = np.append(idArray, '2')

            idArray.append(tuple([2, k]))
        elif i == 3:
            #idArray = np.append(idArray, '12')
            idArray.append(tuple([12, k]))
        elif i == 4:
            #idArray = np.append(idArray, '3')

            idArray.append(tuple([3, k]))
        elif i == 6:
            idArray.append(tuple([23, k]))

        elif i == 8:
            #idArray = np.append(idArray, '4')
            idArray.append(tuple([4, k]))
        k += 1
    #return idArray.astype(int)
    return idArray
############

# File directory
#filePath = "rssiData/rsbits"
#allFiles = glob.glob(filePath + "/*rssi_1.csv")
filePath = "gestureData"
#allFiles = glob.glob(filePath + "/subject_ambuj/swipe_ambuj_l1_final.csv")
#allFiles = glob.glob(filePath + "/block_ambuj_l1_final.csv")
allFiles = glob.glob(filePath + "/block_l1.csv")

# Read all files and create dataset
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
rssiData = pd.concat(list_)

meanRssi = -105 # Hardcoded

# Parameters
sample_period = 1e-3
bitDuration = 0.01 # seconds
nbits = 4

# Generate Preamble sequence
samplesPerBit = int(bitDuration/sample_period)
samplesPerBit = 10
preamble = np.array([0, 1, 1, 0])
preambleSequence = generatePreambleSequence(preamble, samplesPerBit)

# r1 = receiver 1, r2 = receiver 2, etc ...
rssiData.columns = ['r1']

# Select RSSI values from each receiver
r1 = np.asarray(rssiData["r1"])

timeFrame = pd.DataFrame()
for column in rssiData:
    rssiVals = rssiData[column]

    # Remove bad samples
    rssi = []
    dropCount = 0
    for i in range(0, len(rssiVals)):
        if rssiVals[i] < -50 and rssiVals[i] > -150:
            rssi.append(rssiVals[i])
        else:
            dropCount += 1

    rssiVals = rssi
    print rssi

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

#rssiData.to_csv("rssivals.csv")

# Extract preamble sequence from received RSSI
sequences = []
for s in KnuthMorrisPratt(rssiVals, preambleSequence): sequences.append(s)

# Adjust for "repeated" preambles
for i in range(0, len(sequences) - 1):
    if (sequences[i] + 1 == sequences[i+1]):
        sequences.pop(sequences[i+1])

# Extract bits
bitStream, compressedBits = extractBits(sequences, preambleSequence, samplesPerBit, nbits)

# Match Bits
refBits = [0, 1, 1, 0, 1, 1, 1, 0]
count = 0
for i in range(0, len(compressedBits)):
    if (compressedBits[i] == refBits):
        count += 1

# BitAccuracy
accuracy = 100*(float(count)/len(compressedBits))
print "Accuracy:", accuracy, "%"

########################################
# Extract Decimal Array
dataBitStream, dataBits = extractDataBits(sequences, preambleSequence, samplesPerBit, nbits)
dataBits = [x for x in dataBits if x != []]
dataBits = np.asarray(dataBits)
dataBits = dataBits.astype(int)
decimalArray = [bool2int(x) for x in dataBits]

# Determine node identifiers (passive Switch)
#identifiers, block_count, swipe_count = getIdentifiers(decimalArray)
ids = getIdentifiers(decimalArray)

## Check if there is a block
#block = []
#for i in ids:
#    if (i[0] == 12):
#        block.append(i)
#
#test = np.zeros(len(decimalArray))
#for i in block:
#    index  = i[1]
#    test[index] = i[0]
###

#times = getTimes(ids, decimalArray)

#np.savetxt('block_l1.csv', ids, delimiter = ',')

# Determine gestures
#if len(block_count) > 0:
#    count = 0
#    block = 0
#    for i in range(1, len(block_count)):
#        if abs(block_count[i-1] - block_count[i]) > 5:
#            print 'block'
#            block += 1
#            count += 1

#if len(swipe_count) > 0:
#    count = 0
#    swipe = 0
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
###########################################

# Plot data
times = np.arange(0, len(decimalArray))
d = {'times': times, 'decimalArray': decimalArray}
df = pd.DataFrame(data=d)
df.to_csv('received_block_l1.csv')
#plt.step(times, decimalArray)

################# Test #######################
#plt.subplot(2,1,1)
#plt.plot(rssi[18500:22000])

#plt.subplot(2,1,2)
#a = np.asarray(timeFrame['r1_time'])
#plt.step(a[18500:22000], rssiVals[18500:22000])
#plt.show()

#plt.subplot(2,1,1)
#plt.plot(rssi[19500:20500])

#plt.subplot(2,1,2)
#plt.plot(rssiVals[19500:20500])
#plt.ylim([-0.2, 1.2])
#plt.show()
#############################################
#rssiVals = rssiVals[19500:20500]
#rssi = rssi[19500:20500]

rssiVals = rssiVals[1:]
d = {'rssi': rssi, 'rssiVals': rssiVals}
df_2 = pd.DataFrame(data=d)
df_2.to_csv('rssi_block_l1.csv')

#np.savetxt('rssi_swipe_l1.csv', np.transpose([rssi, rssiVals]))
#np.savetxt('received_swipeID_l1.csv', ids, delimiter = ',')
#plt.scatter(times, decimalArray)
plt.plot(times, decimalArray)
#plt.step(timeFrame['r1_time'][5000:30000], rssiData['r1'][5000:30000])
#plt.step(timeFrame['r1_time'], rssiData['r1'])
#plt.step(timeFrame['r1_time'][0:5000], rssiData['r1'][0:5000])
plt.ylim([-0.2, 1.2])
