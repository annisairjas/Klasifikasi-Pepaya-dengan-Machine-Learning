import numpy as np 
import cv2 
import os
import re
# -------------------- Utility function ------------------------
def normalize_label(str_):
    str_ = str_.replace(" ", "")
    str_ = str_.translate(str_.maketrans("","", "()"))
    str_ = float(str_.split("-")[1])
    return str_

def normalize_desc(folder, sub_folder):
    text = folder + " - " + sub_folder 
    text = re.sub(r'\d+', '', text)
    text = text.replace(".", "")
    text = text.strip()
    return text

def print_progress(val, val_len, folder, sub_folder, filename, bar_size=10):
    progr = "#"*round((val)*bar_size/val_len) + " "*round((val_len - (val))*bar_size/val_len)
    if val == 0:
        print("", end = "\n")
    else:
        print("[%s] folder : %s/%s/ ----> file : %s" % (progr, folder, sub_folder, filename), end="\r")

# -------------------- Load Dataset ------------------------
 
dataset_dir = "DATASET/" 

imgs = [] #list image matrix 
labels = []
descs = []
for folder in os.listdir(dataset_dir):
    for sub_folder in os.listdir(os.path.join(dataset_dir, folder)):
        sub_folder_files = os.listdir(os.path.join(dataset_dir, folder, sub_folder))
        len_sub_folder = len(sub_folder_files) - 1
        for i, filename in enumerate(sub_folder_files):
            img = cv2.imread(os.path.join(dataset_dir, folder, sub_folder, filename))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
            area2 = cv2.countNonZero(thresh)
            area3 = area2*0.001
            resize = cv2.resize(gray, (0,0), fx=0.3, fy=0.3)
            imgs.append(area3)
            labels.append(normalize_label(os.path.splitext(filename)[0]))
            descs.append(normalize_desc(folder, sub_folder))
            print_progress(i, len_sub_folder, folder, sub_folder, filename)

def calc_luas_all_agls(img, label):
    feature = []
    feature.append(img)
    feature.append(label) 
    return feature

properties = ['histogram']
luas_all_agls = []
for img, label in zip(imgs, labels):  
    luas_all_agls.append(calc_luas_all_agls(img, label))
properties.append("label")

import pandas as pd 
histo_df = pd.DataFrame(luas_all_agls, columns = properties)
histo_df.to_csv("luas_pepaya_dataset.csv")

