import os
import cv2
import numpy as np

def fit_in_digits(size, number):
    changed = str(number)
    while len(changed) < size:
        changed = '0' + changed
    return changed

def make_file_for_classification():
    # Make a dataset directory from the original directory
    # The dataset directory have two types of files
    # The first one is a text file that have a number of lines
    # Each line consists of a file name and labels which present in the image
    # named in the file name
    # Output: a dataset directory named in "dataset"
    root_path = os.path.abspath('..')
    original_path = os.path.join(root_path, 'data', 'original')
    labels_list = os.listdir(original_path)
    print("labels_list: {}".format(labels_list))
    tuple_list = []
    for labels in labels_list:
        filename_list = os.listdir(os.path.join(original_path, labels))
        print("labels: {} - filename_list: {}".format(labels, filename_list))
        for filename in filename_list:
            if filename.find('CT') != -1:
                tuple_list.append([filename, labels.replace('-', ','), cv2.imread(os.path.join(original_path, labels, filename))])
    print("tuple_list: {}".format(tuple_list))
    num = 1
    for tuple in tuple_list:
        tuple[0] = 'n' + fit_in_digits(4, num) + '.jpg'
        num += 1
    print("fitted tuple_list: {}".format(tuple_list))
    with open(os.path.join(root_path, 'data', 'dataset', 'labels.txt'), 'w') as f:
        for tuple in tuple_list:
            f.write(''.join([tuple[0], '\t', tuple[1], '\n']))
            cv2.imwrite(os.path.join(root_path, 'data', 'dataset', tuple[0]), tuple[-1])
    print("ended...")

def partition_into_tt(path, ratio):
    """

    :param path: a string
    :param ratio: a float
    :return:
    """
    """
        dataset: a list of tuples which consist of filename, labels, object, [[filename, labels, object], ...]
            filename: a string, an image name
            labels: a string, labels present in the image
            object: a numpy, an image
    """
    dataset = []
    with open(os.path.join(path, 'labels.txt'), 'r') as f:
        for line in f.readlines():
            line = line[:-1].split('\t')
            dataset.append([line[0], line[1], cv2.imread(os.path.join(path, line[0]))])
    """
        dicted: a dictionary of tuples which a key is labels and a corresponding values are a list,
                {labels: [[filename, object], ...], ...}
            labels: a string
            filename: a string
            object: a numpy
    """
    dicted = {}
    for data in dataset:
        if dicted.get(data[1]) == None:
            dicted[data[1]] = []
        dicted[data[1]].append([data[0], data[-1]])
    num_dicted = 0
    for key in dicted.keys():
        num_dicted += len(dicted[key])
        print("dicted - key: {} - len: {}".format(key, len(dicted[key])))
    print("dicted - # dicted: {}".format(num_dicted))
    """
        train: a dictionary ...
        test: a dictionary ...
    """
    train = {}
    test = {}
    for key in dicted.keys():
        num = int(ratio * len(dicted[key]))
        train[key] = []
        while len(train[key]) <= num:
            p = np.random.randint(len(dicted[key]))
            train[key].append(dicted[key][p])
            dicted[key].remove(dicted[key][p])
        test[key] = []
        while len(dicted[key]) > 0:
            test[key].append(dicted[key][-1])
            dicted[key].remove(dicted[key][-1])
    num_train = 0
    for key in train.keys():
        num_train += len(train[key])
        print("train - keys: {} - len: {}".format(key, len(train[key])))
    print("train - # train: {}".format(num_train))
    num_test = 0
    for key in test.keys():
        num_test += len(test[key])
        print("test - keys: {} - len: {}".format(key, len(test[key])))
    print("test - # test: {}".format(num_test))
    #
    train_path = os.path.join(os.path.abspath(os.path.join(path, '..')), 'train')
    with open(os.path.join(train_path, 'labels.txt'), 'w') as f:
        for key in train.keys():
            for tp in train[key]:
                f.write(''.join([tp[0], '\t', key, '\n']))
                cv2.imwrite(os.path.join(train_path, tp[0]), tp[-1])
    test_path = os.path.join(os.path.abspath(os.path.join(path, '..')), 'test')
    with open(os.path.join(test_path, 'labels.txt'), 'w') as f:
        for key in test.keys():
            for tp in test[key]:
                f.write(''.join([tp[0], '\t', key, '\n']))
                cv2.imwrite(os.path.join(test_path, tp[0]), tp[-1])