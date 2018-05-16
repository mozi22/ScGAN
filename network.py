import tensorflow as tf
import lmbspecialops as sops
import numpy as np

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
    # return tf.layers.conv2d(
    #     inputs=inputs,
    #     filters=filters,
    #     kernel_size=kernel_size,
    #     strides=stride,
    #     padding='same',
    #     activation=activation,
    #     name=name+'x'
    # )


def _upsample_prediction(inp, num_outputs):
    """Upconvolution for upsampling predictions
    
    inp: Tensor 
        Tensor with the prediction
        
    num_outputs: int
        Number of output channels. 
        Usually this should match the number of channels in the predictions
    """
    output = tf.layers.conv2d_transpose(
        inputs=inp,
        filters=num_outputs,
        kernel_size=4,
        strides=2,
        padding='same',
        activation=None,
        name="upconv"
    )
    return output

def _predict_flow(inp):
    """Generates a tensor for optical flow prediction
    
    inp: Tensor

    predict_confidence: bool
        If True the output tensor has 4 channels instead of 2.
        The last two channels are the x and y flow confidence.
    """

    

    tmp = tf.layers.conv2d(
        inputs=inp,
        filters=24,
        kernel_size=3,
        strides=1,
        padding='same',
        name='conv1_pred_flow',
        activation=myLeakyRelu
    )

    output = tf.layers.conv2d(
        inputs=tmp,
        filters=2,
        kernel_size=3,
        strides=1,
        padding='same',
        name='conv2_pred_flow',
        activation=None
    )

    
    return output

def _refine(inp, num_outputs, upsampled_prediction=None, features_direct=None,name=None):
    """ Generates the concatenation of 
         - the previous features used to compute the flow/depth
         - the upsampled previous flow/depth
         - the direct features that already have the correct resolution

    inp: Tensor
        The features that have been used before to compute flow/depth

    num_outputs: int 
        number of outputs for the upconvolution of 'features'

    upsampled_prediction: Tensor
        The upsampled flow/depth prediction

    features_direct: Tensor
        The direct features which already have the spatial output resolution
    """
    upsampled_features = tf.layers.conv2d_transpose(
        inputs=inp,
        filters=num_outputs,
        kernel_size=4,
        strides=2,
        padding='same',
        activation=myLeakyRelu,
        name="upconv"
    )


    upsampled_features = tf.contrib.layers.batch_norm(upsampled_features, 
                                        is_training=True, 
                                        epsilon=1e-5, 
                                        decay = 0.9,  
                                        updates_collections=None, 
                                        scope='bn1')

    if num_outputs <= 3:
        act1 = tf.nn.tanh(upsampled_features, name='act1')
    else:
        act1 = tf.nn.relu(upsampled_features, name='act1')



    inputs = [act1, features_direct, upsampled_prediction]
    concat_inputs = [ x for x in inputs if not x is None ]

    return tf.concat(concat_inputs, axis=3)


def change_nans_to_zeros(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x), x)


def conv_down(input_image):
        conv0 = convrelu2(name='conv0', inputs=input_image, filters=16, kernel_size=5, stride=1,activation=myLeakyRelu)
        conv1 = convrelu2(name='conv1', inputs=conv0, filters=32, kernel_size=5, stride=2,activation=myLeakyRelu)

        conv2 = convrelu2(name='conv2', inputs=conv1, filters=64, kernel_size=3, stride=2,activation=myLeakyRelu)

        conv3 = convrelu2(name='conv3', inputs=conv2, filters=128, kernel_size=3, stride=2,activation=myLeakyRelu)
        conv3_1 = convrelu2(name='conv3_1', inputs=conv3, filters=128, kernel_size=3, stride=1,activation=myLeakyRelu)

        conv4 = convrelu2(name='conv4', inputs=conv3_1, filters=256, kernel_size=3, stride=2,activation=myLeakyRelu)
        conv4_1 = convrelu2(name='conv4_1', inputs=conv4, filters=256, kernel_size=3, stride=1,activation=myLeakyRelu)

        return conv4_1, conv3_1, conv2, conv1, conv0

def generator(input, random_dim, is_train, reuse=False,image_pair=None):

    conv4_downsample,conv3_downsample,conv2_downsample,conv1_downsample,conv0_downsample = conv_down(image_pair)


    c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32 # channel num
    s4 = 7
    s42 = 12
    output_dim = 2  # RGB image


    with tf.variable_scope('generator') as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable('w1', shape=[random_dim, s4 * s42 * c4 ], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape=[c4 * s4 * s42], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')

        conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s42, c4], name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn1')
        act1 = tf.nn.relu(bn1, name='act1')

        with tf.variable_scope('refine4'):
            concat4 = _refine(
                inp=act1,
                num_outputs=512,
                upsampled_prediction=None,
                features_direct=conv4_downsample,
                name='upconv1'
            )

        with tf.variable_scope('refine3'):
            concat3 = _refine(
                inp=concat4,
                num_outputs=256,
                upsampled_prediction=None,
                features_direct=conv3_downsample,
                name='upconv2'
            )

        with tf.variable_scope('refine2'):
            concat2 = _refine(
                inp=concat3,
                num_outputs=128,
                upsampled_prediction=None,
                features_direct=conv2_downsample,
                name='upconv2'
            )

        with tf.variable_scope('refine1'):
            concat1 = _refine(
                inp=concat2,
                num_outputs=64,
                upsampled_prediction=None,
                features_direct=conv1_downsample,
                name='upconv2'
            )

        with tf.variable_scope('refine0'):
            concat0 = _refine(
                inp=concat1,
                num_outputs=2,
                upsampled_prediction=None,
                features_direct=conv0_downsample,
                name='upconv2'
            )

        prediction = _predict_flow(concat0)
        return prediction



def discriminator(input, is_train, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        conv0 = convrelu2(name='conv0', inputs=input, filters=32, kernel_size=5, stride=2,activation=myLeakyRelu)
        conv0_b = tf.layers.batch_normalization(conv0)
        conv0_r =myLeakyRelu(conv0_b)


        conv1 = convrelu2(name='conv1', inputs=conv0_r, filters=64, kernel_size=5, stride=2,activation=myLeakyRelu)
        conv1_b = tf.layers.batch_normalization(conv1)
        conv1_r =myLeakyRelu(conv1_b)

        conv2 = convrelu2(name='conv2', inputs=conv1_r, filters=128, kernel_size=5, stride=2,activation=myLeakyRelu)
        conv2_b = tf.layers.batch_normalization(conv2)
        conv2_r =myLeakyRelu(conv2_b)

        conv3 = convrelu2(name='conv3', inputs=conv2_r, filters=256, kernel_size=5, stride=2,activation=myLeakyRelu)
        conv3_b = tf.layers.batch_normalization(conv3)
        conv3_r =myLeakyRelu(conv3_b)

        conv4 = convrelu2(name='conv4', inputs=conv3_r, filters=512, kernel_size=5, stride=2,activation=myLeakyRelu)
        conv4_b = tf.layers.batch_normalization(conv4)
        conv4_r =myLeakyRelu(conv4_b)


        dim = int(np.prod(conv4_r.get_shape()[1:]))
        fc1 = tf.reshape(conv4_r, shape=[-1, dim], name='fc1')
      
        
        w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

        # wgan just get rid of the sigmoid
        logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')

        # logits = tf.nn.sigmoid(logits)

        # dcgan
        return tf.nn.sigmoid(logits), logits, conv3_r
