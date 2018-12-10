from matplotlib import pyplot as plt
import numpy as np

# Data to pyplot
n_groups = 5
accuracy_gt = (100, 100, 100, 100, 100)
accuracy_exp = (50, 50, 50, 50, 50)

# Create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

objects = ('Swipe', 'nTaps', 'Swirl', 'Push', 'Pull')

rects1 = plt.bar(index, accuracy_gt, bar_width,
                 alpha = opacity,
                 color = 'black',
                 label = 'Ground truth')

rects1 = plt.bar(index + bar_width, accuracy_exp, bar_width,
                 alpha = opacity,
                 color = 'red',
                 label = 'Experiments')

plt.xlabel('Hand Gestures')
plt.ylabel('Accuracy (%)')
plt.xticks(index + bar_width/2., objects)
plt.legend()

plt.tight_layout()
plt.show()
