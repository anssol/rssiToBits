import numpy as np
import matplotlib.pyplot as plt
import csv
import sklearn
import pandas as pd


filename = "swipe_l1.csv"
# Collect data
with open(filename, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            inputm = np.append(inputm, row)
        inputm = map(float, inputm)
