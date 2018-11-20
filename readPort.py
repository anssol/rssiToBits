import serial
import time
import numpy as np
from matplotlib import pyplot as plt

#zero_baudrate = 230400
zero_baudrate = 115200
#zero_port = '/dev/ttyACM1'
zero_port = '/dev/ttyACM0'

serial = serial.Serial(port=zero_port, baudrate = zero_baudrate)
serial.timeout = 2

plt.ion()
dataArray = np.array([])

print serial.is_open 
if serial.is_open:
    while True:
        try:
            size = serial.inWaiting()
            if size:
                data = serial.read(size)
                data = data.replace('\n', ' ').replace('\r', '')
                data = data.split()
                data = np.asarray(data)
                data = data.astype(np.float)
                data = data[np.where(data < -30)[0]]
                data = data[np.where(data > -300)[0]]
                print len(data)
                #print data
                x = np.arange(0, len(data))
                plt.plot(x, data)
                plt.draw()
                plt.pause(0.01)
                plt.clf()
            else:
                print 'no data'
            time.sleep(1)
        except ValueError:
            continue
else:
    print 'serial is not open'
