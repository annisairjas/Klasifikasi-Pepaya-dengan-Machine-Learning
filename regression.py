import numpy as np
import cv2 as cv
import math
import imutils.paths as path
from tqdm import tqdm
import pandas as pd

datapepaya = pd.read_csv('histo_pepaya_dataset.csv')
datapepaya.head()

from sklearn.model_selection import train_test_split
X=datapepaya[['histogram']].values
Y=datapepaya['label'].values

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = \
                    train_test_split(X, Y, test_size=0.25, random_state=42)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression().fit(train_x, train_y)

import pickle
filename = 'reg_model.sav'
pickle.dump(lin_reg, open(filename, 'wb'))

pred_ynn=lin_reg.predict(test_x)
print(pred_ynn)
print(lin_reg.intercept_)
print(lin_reg.coef_)
print(lin_reg.score(test_x, test_y))


