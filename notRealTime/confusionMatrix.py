import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i,j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

ref = 20.

# Ambuj
ambuj_swipe_l1 = 19 
ambuj_block_l1 = 19

array = [[19/20., 19/20., 0, 0]]
df_cm = pd.DataFrame(array, index = [i for i in "SwipeBlockTapsSwirl"],
                     columns = [i for i in "SwipeBlockTapsSwirl"])

#plt.figure(figsize = (10, 7))
#sn.heatmap(df_cm, annot = True)

