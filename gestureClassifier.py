import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import pickle
import itertools

# TODO: Remove zeros from dataset!

# to debug
from sklearn import datasets

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


# TODO:
# 1. Import the CSV file 
# 2. Label the data
# 3. Implement a classifier
#swipe = pd.read_csv('trainingData/swipe.csv')
#push = pd.read_csv('trainingData/push.csv')
#pull = pd.read_csv('trainingData/pull.csv')

# Merge DataFrames
data_train = pd.concat([swipe, push, pull])
data_train = shuffle(data_train)

duration = data_train['duration']
dropTime = data_train['dropTime']
riseTime = data_train['riseTime']
#vals = trainingData['vals']

# ---------------------------------------------------------------
# Create labels (Add more gestures later)
# 0 = Nothing (do we need to classify this?)
# 1 = Swipe
# 2 = Push
# 3 = Pull
# ---------------------------------------------------------------
labels = np.zeros(len(data_train), dtype=int)

# Swipe
swipeLabels = np.where(duration <= 4)[0]
#sl1 = np.where(np.logical_and(dropTime > 0, dropTime < 8))[0]
#sl2 = np.where(np.logical_and(riseTime > 0, riseTime < 8))[0]
#swipeLabels = np.intersect1d(sl1, sl2)
labels[swipeLabels] = 1

# Push
pushLabels = np.where(dropTime >= 3)[0]
labels[pushLabels] = 2

# Pull
pullLabels = np.where(riseTime >= 4)[0]
labels[pullLabels] = 3
# ---------------------------------------------------------------

# Create a dataframe with labels and append it to existing data
data_train.__delitem__("vals")
data_train.__delitem__(data_train.columns[0])
#data_train.__delitem__("duration")

# Make a matrix of training data
X = []
for row in data_train.iterrows():
    index, data = row
    X.append(data.tolist())
y = labels.tolist()

# Divide X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# Train the classifier using SVM (one against one)
svm_model_1 = svm.SVC(decision_function_shape = 'ovo').fit(X_train,y_train)
svm_model_2 = svm.SVC(kernel = 'linear', C = 1).fit(X_train, y_train)

# SVM (one against all)
svm_model_3 = svm.LinearSVC().fit(X_train, y_train)

# KNN does not work
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train)

# Predict the output using the test data
predict_1 = svm_model_1.predict(X_test)
predict_2 = svm_model_2.predict(X_test)
predict_3 = svm_model_3.predict(X_test)

# Model accuracy for X test
accuracy_1 = svm_model_1.score(X_test, y_test)
accuracy_2 = svm_model_2.score(X_test, y_test)
accuracy_3 = svm_model_3.score(X_test, y_test)

# Bad
accuracy_knn = knn.score(X_test, y_test)

# Save models to file
filename = 'classifiers/svm_model_1.sav'
pickle.dump(svm_model_2, open(filename, 'wb'))

# Confusion matrix (" Do this in another file")
cm = confusion_matrix(y_test, predict_2)
np.set_printoptions(precision=2)
plt.figure()
target_names = np.array(["Nothing", "Swipe", "Push", "Pull"])
plot_confusion_matrix(cm, classes=target_names, title="Confusion matrix")
plt.show()

