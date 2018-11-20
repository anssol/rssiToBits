#!/bin/bash

#python read-rssi.py > /dev/null 2>&1 
#mkfifo rssivals
python read-rssi.py
cat /dev/ttyACM0 > rssivals 

