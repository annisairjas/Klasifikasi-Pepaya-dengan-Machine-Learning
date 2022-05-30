import numpy as np
import cv2 as cv
import imutils.paths as path
from tqdm import tqdm
import pandas as pd

datapepaya = pd.read_csv('glcm_pepaya_dataset.csv')
datapepaya.head()

from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

X=datapepaya[['energy_0','contrast_0','dissimilarity_0','ASM_0',
                    'energy_45','contrast_45','dissimilarity_45','ASM_45',
                    'energy_90','contrast_90','dissimilarity_90','ASM_90',
                    'energy_135','contrast_135','dissimilarity_135','ASM_135']].values

labelnya=datapepaya['label']

le = LabelEncoder()
le.fit(datapepaya["label"].values)
print(" categorical label : \n", le.classes_)

Y = le.transform(datapepaya['label'].values)
Y = to_categorical(Y)

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = \
                    train_test_split(X, Y, test_size=0.25, random_state=42)

import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

#KNN
from sklearn.neighbors import KNeighborsClassifier
KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski', metric_params=None,n_jobs=None,n_neighbors=3,p=2,weights='uniform')

neigh=KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_x,train_y)

filename = 'finalized_model5.sav'
pickle.dump(neigh, open(filename, 'wb'))

pred_ynn=neigh.predict(test_x)
score=accuracy_score(test_y,pred_ynn)
print(score)
scores = cross_val_score(neigh, X, Y, cv=5)
print(scores)
print('cv_scores mean:{}'.format(np.mean(scores)))