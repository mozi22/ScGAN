import tensorflow as tf
import lmbspecialops as sops


def myLeakyRelu(x):
    """Leaky ReLU with leak factor 0.1"""
    # return tf.maximum(0.1*x,x)
    return sops.leaky_relu(x, leak=0.2)

def convrelu2(name,inputs, filters, kernel_size, stride, activation=None):

    tmp_y = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=[kernel_size,1],
        strides=[stride,1],
        padding='same',
        name=name+'y',
        activation=activation
    )


    tmp_x = tf.layers.conv2d(
        inputs=tmp_y,
        filters=filters,
        kernel_size=[1,kernel_size],
        strides=[1,stride],
        padding='same',
        activation=activation,
        name=name+'x'
    )

    return tmp_x


def train_network(image_pair):

    # contracting part
    with tf.variable_scope('down_convs'):


        conv1 = convrelu2(name='conv0', inputs=image_pair, filters=16, kernel_size=5, stride=2,activation=myLeakyRelu)
        conv1_1 = convrelu2(name='conv1', inputs=conv0, filters=16, kernel_size=5, stride=1,activation=myLeakyRelu)

        conv2 = convrelu2(name='conv2', inputs=conv1, filters=32, kernel_size=3, stride=2,activation=myLeakyRelu)
        conv2_1 = convrelu2(name='conv2', inputs=conv2, filters=32, kernel_size=3, stride=1,activation=myLeakyRelu)

        conv3 = convrelu2(name='conv3', inputs=conv2_1, filters=64, kernel_size=3, stride=2,activation=myLeakyRelu)
        conv3_1 = convrelu2(name='conv3_1', inputs=conv3, filters=64, kernel_size=3, stride=1,activation=myLeakyRelu)

        conv4 = convrelu2(name='conv4', inputs=conv3_1, filters=128, kernel_size=3, stride=2,activation=myLeakyRelu)
        conv4_1 = convrelu2(name='conv4_1', inputs=conv4, filters=128, kernel_size=3, stride=1,activation=myLeakyRelu)


        # dense layer

        dense_slice_shape = conv4_1.get_shape().as_list()
        dense_slice_shape[-1] = 96
        units = 1
        for i in range(1,len(dense_slice_shape)):
            units *= dense_slice_shape[i]

        dense5 = tf.layers.dense(
                tf.contrib.layers.flatten(tf.slice(conv4_1, [0,0,0,0], dense_slice_shape)),
                units=units,
                activation=myLeakyRelu,
                name='dense5'
                )

        # perform classification.
        logits = tf.layers.dense(inputs=dense5, units=1, activation=tf.nn.sigmoid)

        return logits


