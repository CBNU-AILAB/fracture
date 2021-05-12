import tensorflow as tf

def relu(name, input):
    """
        ReLU activation function

    :param name: a string
    :param input: a 4D tensor
    :return:
    """
    with tf.variable_scope(name):
        return tf.nn.relu(input)

def batch_norm(name, input, training=True):
    """
        Batch normalization layer

    :param name: a string
    :param input: a tensor, [batch_size, ...]
    :param training: 
    :return: 
    """
    with tf.variable_scope(name):
        return tf.layers.batch_normalization(input, training=training)

def maxpool2d(name, input, ksz, stride, padding):
    """
        Max pooling layer for 2D images
    
    :param name: a string
    :param input: a 4D tensor
    :param ksz: an integer, the size of filter
    :param stride: an integer
    :param padding: a string, all string need to be uppercase
    :return:
    """
    with tf.variable_scope(name):
        k_shapes = [1] * 4
        k_shapes[1:3] = [ksz] * 2
        s_shapes = [1] * 4
        s_shapes[1:3] = [stride] * 2
        return tf.nn.max_pool(input, k_shapes, s_shapes, padding)


def avgpool2d(name, input, ksz, stride, padding):
    """
        Average pooling layer for 2D images

    :param name: a string
    :param input: a 4D tensor
    :param ksz: an integer, the size of filter
    :param stride: an integer
    :param padding: a string, all strings need to be uppercase
    :return:
    """
    with tf.variable_scope(name):
        k_shapes = [1] * 4
        k_shapes[1:3] = [ksz] * 2
        s_shapes = [1] * 4
        s_shapes[1:3] = [stride] * 2
        return tf.nn.avg_pool(input, k_shapes, s_shapes, padding)

def conv2d(name, input, filters, ksz, stride, padding):
    """
        Convolution layer for 2D images

    :param name: a string
    :param input: a 4D tensor
    :param filters: an integer, # filters
    :param ksz: an integer, the size of filters
    :param stride: an integer
    :param padding: a string, all strings need to be uppercase
    :return:
    """
    with tf.variable_scope(name):
        i_shapes = input.get_shape().as_list()
        k_shapes = [1] * 4
        k_shapes[1:3] = [ksz] * 2
        s_shapes = [1] * 4
        s_shapes[1:3] = [stride] * 2
        W = tf.get_variable('W', shape=[k_shapes[1], k_shapes[2], i_shapes[3], filters], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', [filters], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        return tf.nn.conv2d(input, filter=W, strides=s_shapes, padding=padding) + b