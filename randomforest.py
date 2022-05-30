import pandas as pd
import numpy as np

datapepaya = pd.read_csv('glcm_pepaya_dataset.csv')
datapepaya.head()

from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

def decimal_scaling(data):
    data = np.array(data, dtype=np.float32)
    max_row = data.max(axis=0)
    c = np.array([len(str(int(number))) for number in np.abs(max_row)])
    return data/(10**c)
X=datapepaya[['energy_0','contrast_0','dissimilarity_0','ASM_0',
                    'energy_45','contrast_45','dissimilarity_45','ASM_45',
                    'energy_90','contrast_90','dissimilarity_90','ASM_90',
                    'energy_135','contrast_135','dissimilarity_135','ASM_135']].values

labelnya=datapepaya['label']

le = LabelEncoder()
le.fit(datapepaya["label"].values)


print(" categorical label : \n", le.classes_)

y = le.transform(datapepaya['label'].values)
y = to_categorical(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

import pickle
filename = 'random_model.sav'
pickle.dump(clf, open(filename, 'wb'))

# prediction on test set
y_pred=clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv=5)
print(scores)
print('cv_scores mean:{}'.format(np.mean(scores)))

