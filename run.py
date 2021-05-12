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
ResizeMethod = image_ops.ResizeMethod
_RESIZE_METHODS = {
    'bilinear': ResizeMethod.BILINEAR,
    'nearest': ResizeMethod.NEAREST_NEIGHBOR,
    'bicubic': ResizeMethod.BICUBIC,
    'area': ResizeMethod.AREA,
    'lanczos3': ResizeMethod.LANCZOS3,
    'lanczos5': ResizeMethod.LANCZOS5,
    'gaussian': ResizeMethod.GAUSSIAN,
    'mitchellcubic': ResizeMethod.MITCHELLCUBIC
}

label_parameter=0.05


def Select_Label(y_true, y_pred, t=label_parameter):
    num_class= y_pred.get_shape()[1]
    values, indices = tf.math.top_k(y_pred, k=3)
    i0 = tf.constant(0)
    m0 = tf.ones([1, num_class])
    c = lambda i, m: i < tf.shape(y_true)[0]
    def test_body(i, m):
        if values[i][0] - values[i][1] > t:
            label = indices[i][:1]
        else :
            if values[i][1] - values[i][2] >t:
                label = indices[i][:-1]
            else :
                label = indices[i]
        one_hot = tf.expand_dims(K.sum(tf.one_hot(label, num_class), 0), axis=0)
        return [i+1, tf.keras.layers.Concatenate(axis=0)([m, one_hot])]
    z=tf.while_loop(
        c, test_body, loop_vars=[i0, m0],
        shape_invariants=[i0.get_shape(), tf.TensorShape([None, num_class])])
    return z[1][1:]
def Custom_Accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, Select_Label(y_true, y_pred)), axis=-1)
def Custom_Hamming_Loss(y_true, y_pred):
    y_pred = Select_Label(y_true, y_pred)
    return K.mean(y_true*(1-y_pred)+(1-y_true)*y_pred)
def Custom_Precision(y_true, y_pred):
    y_pred_yn = K.round(K.clip(Select_Label(y_true, y_pred), 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
    y_true_yn = K.round(K.clip(y_true, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    count_true_positive = K.sum(y_true_yn * y_pred_yn)
    count_true_positive_false_positive = K.sum(y_pred_yn)
    return count_true_positive / (count_true_positive_false_positive + K.epsilon())
def Custom_Recall(y_true, y_pred):
    y_pred_yn = K.round(K.clip(Select_Label(y_true, y_pred), 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
    y_true_yn = K.round(K.clip(y_true, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    count_true_positive = K.sum(y_true_yn * y_pred_yn)
    count_true_positive_false_positive = K.sum(y_true_yn)
    return count_true_positive / (count_true_positive_false_positive + K.epsilon())
def label_split(df):
    df["tags"]=df["tags"].apply(lambda x:x.split(","))
    return df


class ResizeAug(tf.keras.layers.Layer):
    def __init__(self, resize_range=[0.8, 1.2], resize_methods='bilinear', **kwargs):
        super(ResizeAug, self).__init__(**kwargs)
        self.resize_range = resize_range
    def call(self, images, training=None):
        if not training:
            return images
        range_size = int(round((self.resize_range[1] - self.resize_range[0]),1 )*10)
        index=np.random.randint(low = 0, high = range_size)

        input_shape = array_ops.shape(images)
        height = tf.dtypes.cast(tf.dtypes.cast(input_shape[1], tf.float32) * (self.resize_range[0]+(index*0.1)), tf.int32)
        width = tf.dtypes.cast(tf.dtypes.cast(input_shape[2], tf.float32) * (self.resize_range[0]+(index*0.1)), tf.int32)
        images  = tf.image.resize(images, [height, width])

        return images
class RotateAug(tf.keras.layers.Layer):
    def __init__(self,  **kwargs):
        super(RotateAug, self).__init__(**kwargs)
    def call(self, images, training=None):
        if not training:
            return images
        angle = np.random.randint(low = 0, high =10)
        images  = tfa.image.rotate(images, angle*3)
        return images




def fracture_model():
    data_augmentation = tf.keras.Sequential([
        ResizeAug(),
        RotateAug(),
        layers.experimental.preprocessing.RandomFlip("vertical"),
        layers.experimental.preprocessing.RandomTranslation(height_factor=[-0.08, 0.08], width_factor=(-0.12, 0.12), fill_mode='nearest'),
        layers.experimental.preprocessing.CenterCrop(height=225, width=225),
        layers.GaussianNoise(0.2),
    ])

    model = tf.keras.Sequential([
        data_augmentation,
        fracture_resnet(),
    ])
    return model


def train_model(data, model):
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=data,
        directory=train_dir+"/train",
        x_col = 'filename',
        y_col= 'tags',
        target_size=(384, 384),
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        classes=['1','2','3','4','5','6','7','8','9','10','11'],)
    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=data,
        directory=train_dir+"/train",
        subset="validation",
        x_col = 'filename',
        y_col= 'tags',
        class_mode='categorical',
        shuffle=True,
        classes=['1','2','3','4','5','6','7','8','9','10','11'],
        target_size=(384, 384),
    )
    model.compile(optimizer = tf.keras.optimizers.Adam(),
                  loss = 'binary_crossentropy',
                  metrics=[Custom_Accuracy, Custom_Precision, Custom_Recall, Custom_Hamming_Loss])
    #fit의 경우 전체 데이터 셋을 사용하기 때문에 적은 양의 데이터 셋에 적합하고, fit_Generator의 경우 multiprocessing을 진행할때, 중복되는 데이터를 없애주기 때문에 대량의 데이터 셋에 적합하다.
    model.fit(train_generator,
              epochs=5,
              verbose=1,
              validation_data = validation_generator,
              )
def cross_valid(df,model, n_splits =3):
    sum = [0]*5
    metric_list =  ['Accuracy', 'Precision', 'Recall', 'Hamming_Loss', 'Loss']
    for i in range(n_splits):
        uniqueTuple= set([tuple(l) for l in df['tags']])
        test_index =[]
        for uniqueLabel in list(uniqueTuple):
            indices = []
            for index, tag in enumerate(df.tags):
                if(tag==list(uniqueLabel)) :
                    indices.append(index)
            for i in np.random.randint(low=0, high=len(indices), size=int(len(indices)*1/n_splits)):
                test_index.append(indices[i])
        train_index = [x for x in range(df.shape[0]) if x not in test_index]
        trainData = df.iloc[train_index]
        testData = df.iloc[test_index]
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=trainData,
            directory=train_dir+"/train",
            x_col = 'filename',
            y_col= 'tags',
            target_size=(384, 384),
            batch_size=12,
            class_mode='categorical',
            shuffle=True,
            classes=['1','2','3','4','5','6','7','8','9','10','11'],)
        validation_generator = validation_datagen.flow_from_dataframe(
            dataframe=testData,
            directory=train_dir+"/train",
            subset="validation",
            x_col = 'filename',
            y_col= 'tags',
            class_mode='categorical',
            shuffle=True,
            classes=['1','2','3','4','5','6','7','8','9','10','11'],
            target_size=(384, 384),
        )
        model.compile(optimizer = tf.keras.optimizers.Adam(),
                      loss = 'binary_crossentropy',
                      metrics=[Custom_Accuracy, Custom_Precision, Custom_Recall, Custom_Hamming_Loss])
        history = model.fit(train_generator,
            epochs=2,
            verbose=1,
            validation_data = validation_generator,
        )
        sum[0] += history.history['Custom_Accuracy'][-1]
        sum[1] += history.history['Custom_Precision'][-1]
        sum[2] += history.history['Custom_Recall'][-1]
        sum[3] += history.history['Custom_Hamming_Loss'][-1]
        sum[4] += history.history['loss'][-1]
    for i, name in enumerate(metric_list):
        print(name + ": ", sum[i]/n_splits)
    return sum

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






train_dir = os.getcwd()+'/dataset/radiograph_excel/train'
test_dir = os.getcwd()+'/dataset/radiograph_excel/test'
df = label_split(pd.read_csv(train_dir+"/train/imageLists.csv"))
df = overSampling(df = df,  classes=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'], N=5, T=5).augment()



train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1, data_format='channels_last' )
validation_datagen = ImageDataGenerator(rescale=1. / 255,  data_format='channels_last' )
test_datagen = ImageDataGenerator(rescale=1. / 255,  data_format='channels_last' )

model = tf.keras.models.Sequential()
# cross_valid(df, fracture_model())
# train_model(df, fracture_model())
train_model(df, model)
# test_generator=test_datagen.flow_from_dataframe(
#     dataframe=testData,
#     directory=train_dir+"/train",
#     x_col = 'filename',
#     y_col= 'tags',
#     shuffle=False,
#     class_mode="categorical",
#     target_size=(384, 384))
#print(model.evaluate_generator(generator=test_generator, verbose=2))

