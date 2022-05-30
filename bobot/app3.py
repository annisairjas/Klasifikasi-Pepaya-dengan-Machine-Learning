from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np
import cv2 as cv
#import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
from skimage.color import rgb2gray
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

import pickle

# Define a flask app
app = Flask(__name__)

# -------------------- Utility function ------------------------


def normalize_label(str_):
    str_ = str_.replace(" ", "")
    str_ = str_.translate(str_.maketrans("","", "()"))
    str_ = str_.split("-")[1]
    return str_

def normalize_desc(folder, sub_folder):
    text = folder + " - " + sub_folder 
    text = re.sub(r'\d+', '', text)
    text = text.replace(".", "")
    text = text.strip()
    return text

def print_progress(val, val_len, folder, sub_folder, img_path, bar_size=10):
    progr = "#"*round((val)*bar_size/val_len) + " "*round((val_len - (val))*bar_size/val_len)
    if val == 0:
        print("", end = "\n")
    else:
        print("[%s] folder : %s/%s/ ----> file : %s" % (progr, folder, sub_folder, img_path), end="\r")

def model_predict(img_path):
    np.set_printoptions(suppress=True)
    
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(0,16), dtype=np.float32)
    imgs = [] #list image matrix 
    labels = []
    descs = []
    #img glcm
    img = cv.imread(img_path)
            
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY)
    area2 = cv.countNonZero(thresh)
    area3 = area2*0.001
                
    imgs.append(area3)
    labels = [0]
    #labels.append(normalize_label(os.path.splitext(img_path)[0]))
    #descs.append(normalize_desc(folder, sub_folder))
                
    #print_progress(img_path)

    from skimage.feature import greycomatrix, greycoprops

    # ----------------- calculate greycomatrix() & greycoprops() for angle 0, 45, 90, 135 ----------------------------------
    def calc_glcm_all_agls(img, label):
    
    
        feature = []
    
        feature.append(img)
        feature.append(label) 
    
        return feature

    # ----------------- call calc_glcm_all_agls() for all properties ----------------------------------
    properties = ['histogram']

    glcm_all_agls = []
    for img, label in zip(imgs, labels):  
        glcm_all_agls.append(
            calc_glcm_all_agls(img, 
                                label)
                            )


    properties.append("label")

    import pandas as pd 

    # Create the pandas DataFrame for GLCM features data
    hsv_df = pd.DataFrame(glcm_all_agls, 
                        columns = properties)

    #save to csv
    hsv_df.to_csv("hsv_pepaya_datainput.csv")

    datapepaya = pd.read_csv('hsv_pepaya_datainput.csv')
    datapepaya.head()

    from sklearn.preprocessing import LabelEncoder
    from keras.utils.np_utils import to_categorical


    from sklearn.model_selection import train_test_split

    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix


    # ------------------------ Data Normalization menggunakan Decimal Scaling --------------------------------

    X=datapepaya[['histogram']].values
    

    #X_tuple = np.vstack(X)
    #X_array = np.asarray(X)
    # Load the image into the array
    #data[0] = X_array
    
    
    # Load the model
    #model = tensorflow.keras.models.load_model('ripeness.h5')
    loaded_model = pickle.load(open('C:/Users/kirus/OneDrive/Documents/ta/pepaya_knn/reg_model.sav', 'rb'))

    # run the inference
    preds = ""
    prediction = loaded_model.predict(X)
    # max_val = np.amax(prediction)*100
    # max_val = "%.2f" % max_val
    #prediksi = (X*(-2.8891969)*10**(-8))+0.99913909442044
    #prediksi2 = map(int,str(prediksi))
    preds = np.array_str(prediction)

    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path)
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)