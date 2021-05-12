import os
import tensorflow as tf

def retrain(configurations):
    """
        Retrain a model saved previously

    :param configurations: a dictionary
    :return:
    """
    with tf.Session() as sess:
        # Restore model information
        saver = tf.train.import_meta_graph(os.path.join(configurations['ckpt_path'], 'model.ckpt.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join(configurations['ckpt_path'])))

        # This will output all the operations defined in the graph. If you want to look at all the names
        # you can change 'str(op)' to 'op.name'. As you can see, the return type of 'get_operations()'
        # is a structured such as JSON.
        # with open(os.path.join(configurations['tmp_path'], 'ops.txt'), 'w') as f:
        #     for op in tf.get_default_graph().get_operations():
        #         f.write(str(op) + '\n')

        # something to do