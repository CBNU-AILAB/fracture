import os
import numpy as np
import tensorflow as tf

from utils.Load import Load
from utils.Batch import Batch
from utils.Preproc import Preproc
from utils.DataAug import DataAug
from utils.loss import sigmoid_cross_entropy_loss

def get_train(configurations):
    """
    :param configurations:
    :return:
    """
    load = Load()
    load.load(configurations['train_path'])
    return load.get()

def get_test(configurations):
    """
    :param configurations:
    :return:
    """
    load = Load()
    load.load(configurations['test_path'])
    return load.get()

def preproc(train_images, test_images):
    """
    :param train_images:
    :param test_images:
    :return:
    """
    pp = Preproc()
    means = pp.get_means(train_images)
    subtracted_train = pp.subtract_in_each_dim(train_images, means)
    subtracted_test = pp.subtract_in_each_dim(test_images, means)
    return subtracted_train, subtracted_test

def one_hot(labels, configurations):
    """

    :param labels: a list of lists of integers
    :param configurations: a dictionary for configurations
    :return labels in one-hot vector: a numpy, NC
    """
    oned = np.zeros((len(labels), configurations['num_classes']), dtype=np.float32)
    for n in range(len(labels)):
        for i in range(len(labels[n])):
            oned[n, labels[n][i]] = 1.0
    return oned

def get_offsets(image, offset_ratios):
    """
    :param image:
    :param offset_ratios:
    :return:
    """
    return [image.shape[0]*offset_ratios[0], image.shape[1]*offset_ratios[1]]

def augment(images, crop_sizes):
    """
    :param images: a numpy, NHWC
    :param crop_sizes: a list of sizes 2, [h, w]
    :return augmented images: a numpy, NHWC

    # 크기 변경(Size Reshape) 80% or 90% or 100% or 110% or 120%
    # 회전(Rotate) 3° or 6° or … or 30°
    # 반전(Flip) False or 좌우반전
    # 이동(Translate) 상하 : -8%∼8%, 좌우 : -12%∼12%
    # 자르기(Crop) 중앙 기반 자르기(Center based Cropping)
    # 잡음(Noise) 점 잡음(Salt and Pepper Noise) 각 기준들을 무작위로 선택하여 사용, 이미지의 크기또한 변경될 수 있음 (모두 다 실행)
    검은색 배경은 RGBD 4개의 채널 중 깊이(Depth)채널에 의한 것이다. 하지만 이 깊이 채널은 데이터 증대보다 먼저 진행되는 전처리 단계에서 제거된다. 따라서
    augmentation된 모든 사진은 학습에 사용, 매 에폭마다 augmentation하여 학습에 사용
    """

    scales = [224, 256, 384, 480, 640]
    flips = [None, 'h']
    degrees = [30 * i for i in range(int(360 / 30))]
    offsets = [[(i-1)/12, (j-1)/12] for i in range(3) for j in range(3)]
    da = DataAug()
    augmented = []
    #translation 은 제외하는게 좋음
    for n in range(images.shape[0]):
        scaled = da.resize(images[n], [scales[np.random.randint(len(scales))]] * 2)
        flipped = da.flip(scaled, 0) if flips[np.random.randint(len(flips))] == 'h' else scaled
        rotated = da.rotate(flipped, degrees[np.random.randint(len(degrees))], [flipped.shape[0]/2, flipped.shape[1]/2])
        translated = da.translate(rotated, get_offsets(rotated, offsets[np.random.randint(len(offsets))]))

        starts = [int((translated.shape[0]-crop_sizes[0])/2), int((translated.shape[1]-crop_sizes[1])/2)]
        sizes = [crop_sizes[0], crop_sizes[1]]
        cropped = da.crop(translated, starts, sizes)

        augmented.append(cropped)
    return np.asarray(augmented)

def resize(images, shapes):
    """
    :param images: a numpy, NHWC
    :param shapes: a list of sizes 2, [h, w]
    :return resized images: a numpy, NHWC
    """
    da = DataAug()
    resized = []
    for n in range(images.shape[0]):
        resized.append(da.resize(images[n], shapes))
    return np.asarray(resized)

def get_acc(probs, labels, threshold):
    """
    :param probs:
    :param labels:
    :param threshold:
    :return:
    """
    first = []
    for n in range(probs.shape[0]):
        i = -1
        v = -1
        for c in range(probs.shape[1]):
            if v < probs[n, c]:
                i = c
                v = probs[n, c]
        first.append(i)
    second = []
    for n in range(probs.shape[0]):
        i = -1
        v = -1
        for c in range(probs.shape[1]):
            if c != first[n] and v < probs[n, c]:
                i = c
                v = probs[n, c]
        second.append(i)
    result = []
    for i in range(len(first)):
        if probs[i, first[i]] - probs[i, second[i]] < threshold:
            result.append([first[i], second[i]])
        else:
            result.append([first[i]])

    for i in range(len(result)):
        result[i].sort()

    label = []
    for n in range(labels.shape[0]):
        ones = []
        for c in range(labels.shape[1]):
            if labels[n, c] > 0.5:
                ones.append(c)
        label.append(ones)

    cors = 0 # number of corrections
    for i in range(len(result)):
        if len(result[i]) == len(label[i]):
            valid = True
            for j in range(len(result[i])):
                valid &= label[i][j] == result[i][j]
            if valid == True:
                cors += 1

    return cors / len(result)

def train(configurations):
    """
    :param configurations: a dictionary
    :return:
    """
    # Load images and labels for trainset and
    # Load images and labels for testset
    train_images, train_labels = get_train(configurations)
    test_images, test_labels = get_test(configurations)

    # Change data type to numpy
    train_images = np.asarray(train_images)
    test_images = np.asarray(test_images)

    # Preprocess all the train images
    train_images, test_images = preproc(train_images, test_images)

    # Make labels to one-hot vector type
    train_labels = one_hot(train_labels, configurations)
    test_labels = one_hot(test_labels, configurations)

    # Batch
    batch_train = Batch(train_images, train_labels, configurations['batch_size'])
    batch_test = Batch(test_images, test_labels, configurations['batch_size'])

    # Define tensors and operations on tensorflow
    x = tf.placeholder(tf.float32, shape=[configurations['batch_size'],
                                          configurations['height'],
                                          configurations['width'],
                                          configurations['num_channels']], name='x')
    y = tf.placeholder(tf.float32, shape=[configurations['batch_size'],
                                          configurations['num_classes']], name='y')

    logits_op = resnet50(x)

    probs_op = tf.nn.sigmoid(logits_op)

    loss_op = sigmoid_cross_entropy_loss(labels=y, logits=logits_op)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        trainer_op = tf.train.AdamOptimizer(learning_rate=configurations['learning_rate']).minimize(loss_op)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        lowest_loss = None

        tf.global_variables_initializer().run()

        for epoch in range(1, configurations['num_epochs']+1):
            train_x, train_y = batch_train.next_to()
            loss, prob, _ = sess.run([loss_op, probs_op, trainer_op], feed_dict={x:augment(train_x, [224, 224]), y:train_y})
            acc = get_acc(prob, train_y, 0.05)
            print("epoch: {} - loss: {} - acc: {}".format(epoch, loss, acc))

            # Save a model that outputs a lowest loss until now
            if lowest_loss == None or lowest_loss > loss:
                print("lowest loss: {}".format(loss))
                lowest_loss = loss
                saver.save(sess, os.path.join(configurations['ckpt_path'], 'model.ckpt'))

            if epoch % configurations['test_cycles'] == 0:
                test_x, test_y = batch_test.next_to()
                loss, prob = sess.run([loss_op, probs_op], feed_dict={x:resize(test_x, [224, 224]), y:test_y})
                acc = get_acc(prob, test_y, 0.05)
                print("In test phase loss: {} - acc: {}".format(loss, acc))


configuration={
    'train_path' : 'C:/Users/hansol/PycharmProjects/fracture/dataset/radiograph_excel/train',
    'test_path' : 'C:/Users/hansol/PycharmProjects/fracture/dataset/radiograph_excel/test',
    'ckpt_path' : 'C:/Users/hansol/PycharmProjects/fracture/save_model',
    'height' : 255,
    'width' : 255,
    'num_channels' : 16,
    'num_classes' : 11,
    'learning_rate' : 0.002,
    'num_epochs' : 50,
    'batch_size' : 24,
    'test_cycles' : 5,
}
train(configuration)