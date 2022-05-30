import pandas as pd
import numpy as np

dataset = pd.read_csv('glcm_pepaya_dataset.csv')
dataset.head()

from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

def decimal_scaling(data):
    data = np.array(data, dtype=np.float32)
    max_row = data.max(axis=0)
    c = np.array([len(str(int(number))) for number in np.abs(max_row)])
    return data/(10**c)
X=decimal_scaling(dataset[['energy_0','contrast_0','dissimilarity_0','ASM_0',
                    'energy_45','contrast_45','dissimilarity_45','ASM_45',
                    'energy_90','contrast_90','dissimilarity_90','ASM_90',
                    'energy_135','contrast_135','dissimilarity_135','ASM_135']].values
)
labelnya=dataset['label']

le = LabelEncoder()
le.fit(dataset["label"].values)


print(" categorical label : \n", le.classes_)

y = le.transform(dataset['label'].values)
#y = to_categorical(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

import pickle
filename = 'svm_model.sav'
pickle.dump(clf, open(filename, 'wb'))

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv=5)
print(scores)
print('cv_scores mean:{}'.format(np.mean(scores)))