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
    #block_count = np.array([])
    swipe_count = []
    k = 0
    for i in decimalArray:
        if i == 1:
            #idArray = np.append(idArray, '1')
            idArray.append(tuple([1, k]))
        elif i == 2:
            #idArray = np.append(idArray, '2')
            #swipe_count = np.append(swipe_count, tuple((2, k)))
            idArray.append(tuple([2, k]))
        elif i == 3:
            i#dArray = np.append(idArray, '12')
            idArray.append(tuple([12, k]))
            #block_count = np.append(block_count, k)
        elif i == 4:
            #idArray = np.append(idArray, '3')
            idArray.append(tuple([3, k]))
        elif i == 8:
            idArray.append(tuple([4, k]))
            #idArray = np.append(idArray, '4')
        k += 1
    return idArray
##########################################3

# File directory
filePath = "gestureData/subject_ambuj"
allFiles = glob.glob(filePath + "/*.csv")

# Read all files and create dataset
list_ = []
inputm = np.array([])
rssiVals = [[]]*len(allFiles)
k = 0
for file_ in allFiles:
    with open(file_, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            inputm = np.append(inputm, row)
        inputm = map(float, inputm)
    print inputm
    rssiVals[k] = inputm
    k = k + 1
    inputm = np.array([])
