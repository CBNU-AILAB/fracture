# More or less standard Python stuff
import datetime
import numpy
import os
import pandas

# Visualisation

import matplotlib.pyplot as plt

# Machine learning
import sklearn
import keras

base_dir = './train/'
color_classes = {'black': 0, 'blue': 1, 'red': 2}
decode_color_classes = {v: k for k, v in color_classes.items()}
num_color_classes = max(color_classes.values()) + 1
type_classes = {'jeans': 0, 'dress': 1, 'shirt': 2}
decode_type_classes = {v: k for k, v in type_classes.items()}
num_type_classes = max(type_classes.values()) + 1

image_files_tuples = []
for dirpath, dirnames, filenames in os.walk(base_dir):
    for filename in filenames:
        filename_full = os.path.join(dirpath, filename)
        labels = os.path.basename(dirpath).split('_')
        color_labels = numpy.zeros((num_color_classes,), numpy.float32)
        color_labels[color_classes[labels[0]]] = 1.
        type_labels = numpy.zeros((num_type_classes,), numpy.float32)
        type_labels[type_classes[labels[1]]] = 1.
        image_files_tuples.append((filename_full, numpy.concatenate((color_labels, type_labels))))
image_files = pandas.DataFrame.from_records(image_files_tuples, columns=['filename', 'targets'])
print('Found ' + str(image_files.shape[0]) + ' annotated images')
