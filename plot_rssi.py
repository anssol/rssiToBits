import numpy as np
import matplotlib.pyplot as plt

n_groups = 3
packetsent=0;


font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 28,
        }


highgainfname='rssi0.txt'
#highgainfname='rssi1.txt'
#highgainfname='rssi2.txt'

filenames = ['rssi_8672_2m_3.txt', 'rssi_8674_2m_3.txt', 'rssi_8686_2m_3.txt', 'rssi_8688_2m_3.txt']

rssi = []
rssis = [[]]*4

count = 0
for i in range(0, len(filenames)):
    with open(filenames[i]) as fp:
          next(fp);
          for line in fp:
            if (int(line) > -120):
              count += 1
              rssi.append(int(line))
              if (count == 50000):
                  rssis[i] = rssi
                  rssi = []
                  count = 0

#plt.plot(rssi)

ax1 = plt.subplot(411)
plt.plot(rssis[0])
plt.setp(ax1.get_xticklabels(), visible=False)
ax2 = plt.subplot(412)
plt.plot(rssis[1])
plt.setp(ax2.get_xticklabels(), visible=False)
ax3 = plt.subplot(413)
plt.plot(rssis[2])
plt.setp(ax3.get_xticklabels(), visible=False)
ax4 = plt.subplot(414)
plt.plot(rssis[3])
plt.show()

#plt.plot(hgtime,hgber, 'r',marker='.', linestyle='-',linewidth=2, label='High gain')

#axes = plt.gca()

#plt.yscale('log')
#axes.set_ylim(ymin=0.00001,ymax=1)
#axes.set_yticks((0.00001,0.0001,0.001,0.01, 0.1,1,2.0))
#axes.set_yticklabels(("0.00001","0.0001","0.001","0.01", "0.1","1"))


#plt.xlabel('Time (second)')
#plt.ylabel('BER')
#plt.legend(loc='lower right',
 #          frameon=False, fontsize=10);

plt.subplots_adjust(left=0.25, bottom=0.17)
plt.show()





