import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from os import getcwd

import os
from tensorflow.keras import layers, losses, Model
from tensorflow.python.ops import array_ops
import matplotlib.pyplot as plt
import cv2
from models.googlenet import GoogLeNet
from models.resnet import fracture_resnet
import shutil
import pandas as pd
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold, cross_val_score
from tensorflow.python.keras.engine.base_preprocessing_layer import PreprocessingLayer
from tensorflow.keras.layers import Lambda, Input
from tensorflow.python.ops import image_ops
import tensorflow_addons as tfa
import heapq

def label_split(df):
    df["tags"]=df["tags"].apply(lambda x:x.split(","))
    return df


class overSampling :
    def __init__(self, df, classes, N, T):
        self.df= df
        self.classes = classes
        self.N = N
        self.T = T

    def get_IRLbl(self):
        count_dict = {}  # 라밸 내부의 key
        df = self.df
        classes= self.classes
        for cls in classes:
            count_dict[cls] = 0
        max = 0
        for index, labels in enumerate(df.tags):
            for cls in classes:
                if cls in labels:
                    count_dict[cls] = count_dict[cls] + 1
        for cls in classes:
            if max < count_dict[cls]:
                max = count_dict[cls]
        IRLbl = {}
        sum = 0
        for cls in classes:
            IRLbl[cls] = max / count_dict[cls]
            sum += IRLbl[cls]
        return IRLbl, sum / len(classes)

    def randomSampling(self, label, ExclusionLabel):
        indices = []
        x_index = []
        df=self.df
        T= self.T
        for index, tag in enumerate(df.tags):
            chk = 0
            for exLabel in ExclusionLabel:
                if (exLabel in tag):
                    chk += 1
                    break
            if label in tag and not chk:
                indices.append(index)
        if len(indices) !=0:
            for i in np.random.randint(low=0, high=len(indices), size=T):
                x_index.append(indices[i])
        return x_index
    def augment(self):
        df = self.df
        image_indices = []
        for k in range(self.N):
            minBag = []
            IRLbl, meanIR = self.get_IRLbl()
            for key, val in IRLbl.items():
                if val > meanIR:
                    minBag.append(key)
            ExclusionLabel = heapq.nsmallest(3, IRLbl, key=IRLbl.get)
            for label in minBag:  # 해당하는 라벨을 포함하는 이미지를 뽑아야함
                d_prime = []
                image_indices.append(self.randomSampling(label, ExclusionLabel))
        indices = list(set([j for sub in image_indices for j in sub]))
        df1=df.append(df.iloc[indices], ignore_index=True)
        return df1
import matplotlib.pyplot as plt





train_dir = os.getcwd()+'/dataset/radiograph_excel/train'
test_dir = os.getcwd()+'/dataset/radiograph_excel/test'
df = label_split(pd.read_csv(train_dir+"/train/imageLists.csv"))
x =['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
t =np.array([1,2,3,4,5,6,7,8,9,10,11])
y= [0]*12
for tag in df.tags :
    for label in x:
         if label in tag :
             y[int(label)]+=1
y_np= np.array(y[1:])
origin_y=y_np/df.shape[0]


y= [0]*12
df = overSampling(df = df,  classes=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'], N=6, T=6).augment()
for tag in df.tags :
    for label in x:
         if label in tag :
             y[int(label)]+=1
y_np= np.array(y[1:])
y=y_np/df.shape[0]
plt.bar(t-0.15, origin_y, color='b', width=0.3)
plt.bar(t+0.15, y, color='r', width=0.3)
for i, v in enumerate(t):
    plt.text(v, y[i]+0.15, round(y[i], 2),                 # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
             fontsize = 9,
             color='red',
             horizontalalignment='center',  # horizontalalignment (left, center, right)
             verticalalignment='bottom')
    plt.text(v, y[i] +0.12, round(origin_y[i], 2),  # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
             fontsize=9,
             color='blue',
             horizontalalignment='center',  # horizontalalignment (left, center, right)
             verticalalignment='bottom')
plt.ylim((0, 0.6))
plt.legend(('origin', 'oversampling'))
plt.show()




