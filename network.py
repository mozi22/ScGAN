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




    inputs = [upsampled_features, features_direct, upsampled_prediction]
    concat_inputs = [ x for x in inputs if not x is None ]

    return tf.concat(concat_inputs, axis=3)


def change_nans_to_zeros(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x), x)


def train_network(image_pair):

    # contracting part
    with tf.variable_scope('generator'):

        with tf.variable_scope('down_convs', reuse=tf.AUTO_REUSE):


            conv0 = convrelu2(name='conv0', inputs=image_pair, filters=16, kernel_size=5, stride=1,activation=myLeakyRelu)
            conv1 = convrelu2(name='conv1', inputs=conv0, filters=32, kernel_size=5, stride=2,activation=myLeakyRelu)
            conv1 = tf.layers.dropout(conv1)

            conv2 = convrelu2(name='conv2', inputs=conv1, filters=64, kernel_size=3, stride=2,activation=myLeakyRelu)
            conv2 = tf.layers.dropout(conv2)

            conv3 = convrelu2(name='conv3', inputs=conv2, filters=128, kernel_size=3, stride=2,activation=myLeakyRelu)
            conv3_1 = convrelu2(name='conv3_1', inputs=conv3, filters=128, kernel_size=3, stride=1,activation=myLeakyRelu)
            conv3_1 = tf.layers.dropout(conv3_1)

            conv4 = convrelu2(name='conv4', inputs=conv3_1, filters=256, kernel_size=3, stride=2,activation=myLeakyRelu)
            conv4_1 = convrelu2(name='conv4_1', inputs=conv4, filters=256, kernel_size=3, stride=1,activation=myLeakyRelu)
            conv4_1 = tf.layers.dropout(conv4_1)

            conv5 = convrelu2(name='conv5', inputs=conv4_1, filters=512, kernel_size=3, stride=2,activation=myLeakyRelu)
            conv5_1 = convrelu2(name='conv5_1', inputs=conv5, filters=512, kernel_size=3, stride=1,activation=myLeakyRelu)
            conv5_1 = tf.layers.dropout(conv5_1)


        # predict flow
        with tf.variable_scope('predict_flow5', reuse=tf.AUTO_REUSE):
            predict_flow4 = _predict_flow(conv5_1)

        with tf.variable_scope('upsample_flow4to3', reuse=tf.AUTO_REUSE):
            predict_flow4to3 = _upsample_prediction(predict_flow4, 3)
            # predict_flow4to3 = change_nans_to_zeros(predict_flow4to3)



        with tf.variable_scope('refine4', reuse=tf.AUTO_REUSE):
            concat4 = _refine(
                inp=conv5_1,
                num_outputs=512,
                upsampled_prediction=predict_flow4to3, 
                features_direct=conv4_1,
                name='paddit'
            )
            # predict_flow_ref4 = _predict_flow(concat4)


        # shape=(8, 20, 32, 384)
        with tf.variable_scope('refine3', reuse=tf.AUTO_REUSE):
            concat3 = _refine(
                inp=concat4, 
                num_outputs=256, 
                features_direct=conv3_1
            )

            predict_flow_ref3 = _predict_flow(concat3)

        # shape=(8, 40, 64, 192)
        with tf.variable_scope('refine2', reuse=tf.AUTO_REUSE):
            concat2 = _refine(
                inp=concat3, 
                num_outputs=128,
                features_direct=conv2
            )
            predict_flow_ref2 = _predict_flow(concat2)

        # shape=(8, 80, 128, 96)
        with tf.variable_scope('refine1', reuse=tf.AUTO_REUSE):
            concat1 = _refine(
                inp=concat2,
                num_outputs=64, 
                features_direct=conv1
            )
            predict_flow_ref1 = _predict_flow(concat1)


        with tf.variable_scope('refine0', reuse=tf.AUTO_REUSE):
            concat0 = _refine(
                inp=concat1,
                num_outputs=32, 
                features_direct=conv0
            )


        with tf.variable_scope('predict_flow2', reuse=tf.AUTO_REUSE):

            predict_flow = _predict_flow(concat0)
        



    return [predict_flow, 
            predict_flow_ref1,
            predict_flow_ref2,
            predict_flow_ref3
            # ,predict_flow_ref4
            ]


def discriminator(input, is_train=True, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        conv0 = convrelu2(name='conv0', inputs=input, filters=32, kernel_size=5, stride=2,activation=None)
        conv0 = tf.layers.batch_normalization(conv0,training=is_train)
        conv0 =myLeakyRelu(conv0)

        conv1 = convrelu2(name='conv1', inputs=conv0, filters=64, kernel_size=3, stride=2,activation=None)
        conv1 = tf.layers.batch_normalization(conv1,training=is_train)
        conv1 =myLeakyRelu(conv1)

        conv2 = convrelu2(name='conv2', inputs=conv1, filters=128, kernel_size=3, stride=2,activation=None)
        conv2 = tf.layers.batch_normalization(conv2,training=is_train)
        conv2 =myLeakyRelu(conv2)

        conv3 = convrelu2(name='conv3', inputs=conv2, filters=256, kernel_size=3, stride=2,activation=None)
        conv3 = tf.layers.batch_normalization(conv3,training=is_train)
        conv3 =myLeakyRelu(conv3)

        conv4 = convrelu2(name='conv4', inputs=conv3, filters=512, kernel_size=3, stride=2,activation=None)
        conv4 = tf.layers.batch_normalization(conv4,training=is_train)
        # conv4_r =myLeakyRelu(conv4_b)

        # dim = int(np.prod(conv3_r.get_shape()[1:]))
        # fc1 = tf.reshape(conv3_r, shape=[-1, dim], name='fc1')
      
        
        # w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
        #                      initializer=tf.truncated_normal_initializer(stddev=0.02))
        # b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
        #                      initializer=tf.constant_initializer(0.0))

        # # wgan just get rid of the sigmoid
        # logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')

        logits = tf.nn.sigmoid(conv4)

        # dcgan
        return logits, conv2 #, acted_out