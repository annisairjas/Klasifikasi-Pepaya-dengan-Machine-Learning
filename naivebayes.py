import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
import imutils.paths as path
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

datapepaya = pd.read_csv('glcm_pepaya_dataset.csv')
datapepaya.head()
datapilih = datapepaya[['energy_0','homogeneity_0','correlation_0','contrast_0',
                    'energy_45','homogeneity_45','correlation_45','contrast_45',
                    'energy_90','homogeneity_90','correlation_90','contrast_90',
                    'energy_135','homogeneity_135','correlation_135','contrast_135']].value_counts()


from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical


from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# ------------------------ Data Normalization menggunakan Decimal Scaling --------------------------------
def decimal_scaling(data):
    data = np.array(data, dtype=np.float32)
    max_row = data.max(axis=0)
    c = np.array([len(str(int(number))) for number in np.abs(max_row)])
    return data/(10**c)


X=decimal_scaling(datapepaya[['energy_0','contrast_0','dissimilarity_0','ASM_0',
                    'energy_45','contrast_45','dissimilarity_45','ASM_45',
                    'energy_90','contrast_90','dissimilarity_90','ASM_90',
                    'energy_135','contrast_135','dissimilarity_135','ASM_135']].values
)
Y=datapepaya['label']



#print("\n\n one hot encoding for sample 0 : \n", Y[0])

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = \
                    train_test_split(X, 
                                     Y, 
                                     test_size=0.25, 
                                     random_state=42)

print("Dimensi data :\n")
print("X train \t X test \t Y train \t Y test")  
print("%s \t %s \t %s \t %s" % (train_x.shape, test_x.shape, train_y.shape, test_y.shape))

from sklearn.naive_bayes import GaussianNB
naive=GaussianNB()
naive.fit(train_x,train_y)

import pickle
filename = 'nb_model.sav'
pickle.dump(naive, open(filename, 'wb'))



pred_ynn=naive.predict(test_x)
print(pred_ynn)

print(test_y)

from sklearn.metrics import accuracy_score
score=accuracy_score(test_y,pred_ynn)
print(score)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(naive, X, Y, cv=5)
print(scores)
print('cv_scores mean:{}'.format(np.mean(scores)))