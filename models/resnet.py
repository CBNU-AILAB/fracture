import tensorflow as tf
from models.residual_block import make_basic_block_layer, make_bottleneck_layer

class ResNetType(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetType, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,kernel_size=(7, 7),strides=2,padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),strides=2,padding="same")
        self.layer1 = make_bottleneck_layer(filter_num=8,blocks=layer_params[0])
        self.layer2 = make_bottleneck_layer(filter_num=16,blocks=layer_params[1],stride=2)
        self.layer3 = make_bottleneck_layer(filter_num=16,blocks=layer_params[2],stride=2)
        self.layer4 = make_bottleneck_layer(filter_num=32,blocks=layer_params[3],stride=2)
        self.layer5 = make_bottleneck_layer(filter_num=32,blocks=layer_params[4], stride=2)
        self.layer6 = make_bottleneck_layer(filter_num=64,blocks=layer_params[5],stride=2)
        self.layer7 = make_bottleneck_layer(filter_num=64, blocks=layer_params[6],stride=2)
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=11, activation=tf.keras.activations.sigmoid)
    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.layer5(x, training=training)
        x = self.layer6(x, training=training)
        x = self.layer7(x, training=training)
        x = self.avgpool(x)
        output = self.fc(x)
        return output







def fracture_resnet():
    return ResNetType(layer_params=[3,1,3,1,5,1,2])


