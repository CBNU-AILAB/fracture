import tensorflow as tf

def sigmoid_cross_entropy_loss(labels, logits):
    """
        Loss with sigmoid cross entropy

        For example, we have labels and logits that each have size of [batch_size, num_classes]. In the each batch,
        we have to sum all the elements calculated by 'sigmoid_cross_entropy...'. After that in the all batch,
        we have to calculate the mean scalar from the vector created by former. So we think that it is:

        TODO: something to do

    :param labels:
    :param logits:
    :return:
    """
    return tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=1))

def warp_loss(labels, logits):
    """
        TODO: something to do

    :param labels:
    :param logits:
    :return:
    """
    return