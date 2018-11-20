import numpy as np
import sys
import select
import os
from matplotlib import pyplot as plt
import pandas as pd
from query import query_yes_no
from pygame import mixer

from sklearn import svm
import pickle

# Import classifier
svm_model = pickle.load(open('classifiers/svm_model_1.sav', 'rb'))

# Initialize mixer for Bell sound
mixer.init()
mixer.music.load('beep.wav')

# Normalizes data to between 0 and 1
def normalize(x):
    return (x - min(x))/(max(x) - min(x))

def movingAverage(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N 

# Retrieves the indexes of rising edges
def getRiseIndexes(signal_rssi, tresh):
    riseIndexes = np.array([], dtype = np.int16)
    for i in range(1, len(signal_rssi)):
        if ((signal_rssi[i] - signal_rssi[i-1]) > tresh):
            riseIndexes = np.append(riseIndexes, i)
    if len(riseIndexes) > 0:
        return riseIndexes

# Retrieves the indexes of falling edges
def getFallIndexes(signal_rssi, tresh):
    fallIndexes = np.array([], dtype = np.int16)
    for i in range(2, len(signal_rssi)):
        if ((signal_rssi[i] - signal_rssi[i-1]) < -tresh):
            fallIndexes = np.append(fallIndexes, i)
    if len(fallIndexes) > 0:
        return fallIndexes

# -----------------------------------------------------------------------------------------
# This algorithm extracts the bits from the duration of a pulse (1's) 
# or the time between pulses (0's) and generates a binary sequence of consequent 1's or 0's.
# Further, it assigns time slots to these bits starting from the rising edge
# of a pulse (1) or the falling edge of a pulse (0). This enables both 0's and
# 1's to later be combined into a full binary sequence by comparison of their time slots.
#
# INPUTS:
# duration: the time duration of a pulse, or the time duration between two pulses
# bit: the bit type to extract (0 or 1)
# t_bit: the time duration of a single bit as defined by the frequency of the oscillator
# edgeSlots: indexes to mark the rising or falling edge of a pulse. This should be a value
#            that represents the correct ordering of the signal.
# -----------------------------------------------------------------------------------------
def split_into_parts(number, n_parts):
    return np.linspace(0, number, n_parts + 1)[1:]
def splitPulseIntoBits(duration, bit, t_bit, edgeSlots):
    nbits = np.round(duration/t_bit) 
    binaryArray = np.array([])
    bitSlots = np.array([])
    index = 0
    for i in nbits:
        # Split the duration and map the result to bits
        splitted = split_into_parts(bit, i) 
        binaryArray = np.append(binaryArray, np.ceil(splitted))
        # Assign incremental time slots to the acquired bits
        bitSlots = np.append(bitSlots, edgeSlots[index] + np.arange(0, len(splitted)))
        index += 1
    return binaryArray.astype(int), bitSlots.astype(int)

# Generates the final sequence of bits by matching the time slots
def generateSequence(bitSlots, bitSlot_0, bitSlot_1, nbits):
    sequence = np.zeros(len(bitSlots))
    # Check for matching 0's and set the bits
    for i in bitSlot_0:
        index_0 = np.where(bitSlots == i)[0][0]
        sequence[index_0] = 0
    # Check for matching 1's and set the bits
    for j in bitSlot_1:
        index_1 = np.where(bitSlots == j)[0][0]
        sequence[index_1] = 1
    # Remove the three unwanted bits due to edge detection of preamble
    return sequence.astype(int)

def convertToDecimal(bitArray):
    decimalVal = 0
    for i in range(0, len(bitArray)):
        decimalVal += bitArray[i]*(2**i)
    return decimalVal

# Extract Gesture Features
def getFeatures(samples, meanVal):
    gesture = np.array([meanVal])

    # Collect the samples at which the gesture occurs
    for i in samples:
        if (i < meanVal):
            gesture = np.append(gesture, i)
    gesture = np.append(gesture, meanVal)

    # Feature 1: Gesture Duration (in samples)
    gestureDuration = len(gesture)
    
    # Features 2,3: Drop Duration and Rise Duration (in samples)
    gestureIndexes = np.arange(0, len(gesture))
    if (len(gestureIndexes) > 0):
        maxDrop = np.where(gesture == min(gesture))[0]
        dropDuration = len(gestureIndexes[0:maxDrop[0]])
        riseDuration = len(gestureIndexes[maxDrop[-1] + 1:])
    else:
        dropDuration, riseDuration = 0, 0
    return gesture, gestureDuration, dropDuration, riseDuration

# Define Sample Period and Frequency of CC1310
sample_freq = 1/250e-6
sample_period = 250e-6

# Time duration of a bit in Seconds
t_bit = 0.1 # S. This defines the bit duration
t_bit = 10e-3

# Open FIFO file
fifo = open("rssivals", "rb")

# Define parameters
#bufferlen = 410
bufferlen = 9000
tresh = 0.3 
N = 20 # Window size of filter
preamble = np.array([0, 1, 1, 0]) 
nbits = 4 # Number of bits
count = 0

# Create DataFrame for training
nPoints = 10
arr = [[]]*nPoints
df = pd.DataFrame(columns = ['duration', 'dropTime', 'riseTime','vals'])
k = 0

rssi = np.array([])
decimalArray = np.array([])
binaryArray = np.array([])
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
    if len(rssi) == bufferlen:

        # Apply filter to signal
        rssi = movingAverage(rssi, N)
        rssivals = []
        for i in range(0, len(rssi)):
            if rssi[i] < -80 and rssi[i] > -130:
                rssivals.append(rssi[i])
        rssi = rssivals

        # Normalize the data
        rssi = normalize(rssi)

        # Replace all RSSI samples by 0 if they are close to the noise floor or 1 if above it
        rssi[rssi < np.max(rssi) - 0.2] = 0
        rssi[rssi > np.min(rssi) + 0.5] = 1

        # Define the duration of a frame
        x = np.arange(0, len(rssi))*sample_period

        # Get the indexes of rising and falling edges         
        riseIndexes = getRiseIndexes(rssi, tresh)
        fallIndexes = getFallIndexes(rssi, tresh)

        # If an event is captured in a single frame, start processing 
        if (type(riseIndexes) and type(fallIndexes) is np.ndarray):
            try:
                # Determine the time between pulses
                zeroWidth = np.round(x[riseIndexes[1:]] - x[fallIndexes[0:-1]], 2)
                
                # HARDCODED: Remove the induced bits from the preamble '0110'
                # Todo: Find a way to make this non-trivial problem more generic 
                if len(zeroWidth > 0):
                    zeroWidth[0] = zeroWidth[0]/2.

                # Determine the duration of the subsequent pulse
                pulseWidth = x[fallIndexes] - x[riseIndexes]
                pulseWidth = pulseWidth[1:len(pulseWidth)]

                # Returns True if there is a pulse following the preamble 
                isPulse = all(x > 0 for x in pulseWidth)

                # Returns True if the time-difference between the pulses are
                # within the available time-slot.  
                timeSlot_ok = all(x <= nbits*t_bit for x in zeroWidth)

                print timeSlot_ok 

#                if isPulse and timeSlot_ok:
                #print "Width ", pulseWidth            
                #print "ZeroWidth ", zeroWidth

                # Extract the time slots of rising edges and falling edges
                fallSlots = riseIndexes[1:] + fallIndexes[0:-1]
                riseSlots = fallIndexes + riseIndexes
                riseSlots = riseSlots[1:len(riseSlots)]

                # Generate sequences of 0's and 1's by dividing the time between pulses or
                # the pulse duration by the time per bit (t_bit)
                binaryArray_0, bitSlots_0 = splitPulseIntoBits(zeroWidth, 0, t_bit, fallSlots)
                binaryArray_1, bitSlots_1  = splitPulseIntoBits(pulseWidth, 1, t_bit, riseSlots)

                # Define the array of all bit slots in order 
                bitSlots = np.sort(np.concatenate((bitSlots_0, bitSlots_1)))

                # Generate the binary sequence from the bit slots
                # This is the signal of interest
                bitSequence = generateSequence(bitSlots, bitSlots_0, bitSlots_1, nbits)

                # Add missing 0's after the last pulse if they exist
                if (nbits - len(bitSequence) > 0):
                    for i in range(0, nbits - len(bitSequence)):
                        bitSequence = np.append(bitSequence, 0)
                
                # Add the preamble to the sequence to differentiate between signals
                binaryArray = np.append(preamble, binaryArray)
                #print "Bit Sequence: ", bitSequence

                # Convert binary array to decimal value
                decimalVal = convertToDecimal(bitSequence)
                if (decimalVal < 16):
                    decimalArray = np.append(decimalArray, decimalVal)
                 #print "Decimal Value: ", decimalVal

                # -----------------------------------------------------------------------
                # Combined real time plots of the RSSI readings and the digitized signal
                # -----------------------------------------------------------------------
                # RSSI
                plt.subplot(2, 1, 1)
                plt.plot(x, rssi)
                plt.ylim([-0.2, 1.2])
                plt.yticks([0,1])
                plt.ylabel("Digitized Signal")
                
                # Digitized signal
                plt.subplot(2, 1, 2)
                x_dec = np.arange(0, len(decimalArray))
                plt.plot(x_dec, decimalArray)
                plt.xticks([])
                plt.ylim([-0.2, 20])
                plt.ylabel("Reconstruction")

                plt.draw()
                # -----------------------------------------------------------------------

            except (ValueError, TypeError):
                continue

        ## Draw plot of RSSI in real time 
        #plt.plot(x, rssi)
        #plt.xlabel('Time [s]', fontsize = 14)
        #plt.ylabel('Normalized RSSI', fontsize = 14)
        ##plt.ylim([-120, -70])
        #plt.ylim([-0.2, 1.2])
        #plt.yticks([0,1])
        #plt.draw()
        #plt.pause(0.01)
        #plt.clf()
        ##print rssi
        ## Draw plot 
        #
        #plt.pause(0.01)

        ## Uncomment later
        plt.pause(0.002)
        plt.clf()

        # Compute mean after 5 iterations 
        if len(decimalArray) == 5:
            mean_vals = np.mean(decimalArray)
        
        # Extract Gesture Features
        if (len(decimalArray) == 30): 
            if (np.asarray([x < mean_vals for x in decimalArray]).any()):
                # Uncomment later
                plt.close()
                gesture, gestureDuration, dropDuration, riseDuration = getFeatures(decimalArray, mean_vals)
                # Create a DataFrame with gesture information
                df = df.append({'duration': gestureDuration, 'dropTime': dropDuration,
                                'riseTime': riseDuration}, ignore_index = True)
                arr[k] = decimalArray
                k = k + 1

                #import scipy.sparse as sparse

                #df = pd.DataFrame(np.arange(1,10).reshape(3,3))
                #arr = sparse.coo_matrix(([1,1,1], ([0,1,2], [1,2,0])),
                #                        shape=(3,3))
                #df['newcol'] = arr.toarray().tolist()
                
                # For visualization/debugging 
                #plt.plot(gesture)
                #plt.show()

                # Try classifier
                sample = np.array([gestureDuration, dropDuration, riseDuration]) 
                sample = sample.reshape(1, -1)
                prediction = svm_model.predict(sample)
                print prediction

                if (len(df) == nPoints):
                    ans = query_yes_no("Do you want to save the data?")
                    if (ans == True):
                        print "Saving to CSV"
                        print ""
                        df.to_csv('trainingData/block.csv')

                        arr = np.asarray(arr)
                        np.savetxt("block_vals.csv", arr, delimiter=",")

                x_dec = np.arange(0, len(decimalArray))
                plt.plot(x_dec, decimalArray)
                plt.xticks([])
                plt.ylim([-0.2, 20])
                plt.ylabel("Reconstruction")
                plt.draw()
                plt.pause(0.002)
                plt.clf()

                #raw_input("Next Gesture. Press Enter to continue...")
                print ""
                print "Next Gesture"
                print ""
                mixer.music.play()
                decimalArray = np.array([])
            else:
                print ""
                print "Next Gesture"
                print ""
                mixer.music.play()
                decimalArray = np.array([])

        # Prepare for next iteration
        rssi = np.array([])
        binaryArray = np.array([])
