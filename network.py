import tensorflow as tf
import lmbspecialops as sops
import numpy as np
import tensorflow.contrib.slim as slim

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
        filters=3,
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

def generator(image_pair, random_dim, is_train, reuse=False):

    with tf.variable_scope('generator'):


        # conv0 = convrelu2(name='conv0', inputs=image_pair, filters=16, kernel_size=5, stride=1,activation=myLeakyRelu)
        conv1 = convrelu2(name='conv1', inputs=image_pair, filters=64, kernel_size=5, stride=2,activation=myLeakyRelu)

        conv2 = convrelu2(name='conv2', inputs=conv1, filters=128, kernel_size=3, stride=2,activation=myLeakyRelu)

        conv3 = convrelu2(name='conv3', inputs=conv2, filters=128, kernel_size=3, stride=2,activation=myLeakyRelu)
        # conv3_1 = convrelu2(name='conv3_1', inputs=conv3, filters=128, kernel_size=3, stride=1,activation=myLeakyRelu)

        conv4 = convrelu2(name='conv4', inputs=conv3, filters=256, kernel_size=3, stride=2,activation=myLeakyRelu)
        # conv4_1 = convrelu2(name='conv4_1', inputs=conv4, filters=256, kernel_size=3, stride=1,activation=myLeakyRelu)

        # conv5 = convrelu2(name='conv5', inputs=conv4_1, filters=512, kernel_size=3, stride=2,activation=myLeakyRelu)
        # conv5_1 = convrelu2(name='conv5_1', inputs=conv5, filters=512, kernel_size=3, stride=1,activation=myLeakyRelu)


    # predict flow
    with tf.variable_scope('predict_flow5'):
        predict_flow4 = _predict_flow(conv4)

    with tf.variable_scope('upsample_flow4to3'):
        predict_flow4to3 = _upsample_prediction(predict_flow4, 2)
        # predict_flow4to3 = change_nans_to_zeros(predict_flow4to3)



    with tf.variable_scope('refine4'):
        concat4 = _refine(
            inp=conv4,
            num_outputs=256,
            upsampled_prediction=predict_flow4to3, 
            features_direct=conv3,
            name='paddit'
        )


    with tf.variable_scope('refine3'):
        concat3 = _refine(
            inp=concat4, 
            num_outputs=128, 
            features_direct=conv2
        )

    with tf.variable_scope('refine2'):
        concat0 = _refine(
            inp=concat3, 
            num_outputs=64,
            features_direct=conv1
        )

    # with tf.variable_scope('refine1'):
    #     concat0 = _refine(
    #         inp=concat2,
    #         num_outputs=128, 
    #         features_direct=conv1
    #     )


    # with tf.variable_scope('refine0'):
    #     concat0 = _refine(
    #         inp=concat1,
    #         num_outputs=32, 
    #         features_direct=conv0
    #     )

    with tf.variable_scope('predict_flow2'):

        predict_flow2 = _predict_flow(concat0)
    
    predict_flow2 = change_nans_to_zeros(predict_flow2)

    return predict_flow4, predict_flow2




# def discriminator(input, is_train, reuse=False):
#     c2, c4, c8, c16 = 64, 128, 256, 512  # channel num: 64, 128, 256, 512
#     with tf.variable_scope('discriminator') as scope:
#         if reuse:
#             scope.reuse_variables()

#         #Convolution, activation, bias, repeat! 
#         conv1 = tf.layers.conv2d(input, c2, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
#                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
#                                  name='conv1')
#         bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')
#         act1 = myLeakyRelu(conv1)
#          #Convolution, activation, bias, repeat! 
#         conv2 = tf.layers.conv2d(act1, c4, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
#                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
#                                  name='conv2')
#         bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
#         act2 = myLeakyRelu(bn2)
#         #Convolution, activation, bias, repeat! 
#         conv3 = tf.layers.conv2d(act2, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
#                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
#                                  name='conv3')
#         bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
#         act3 = myLeakyRelu(bn3)
#          #Convolution, activation, bias, repeat! 
#         conv4 = tf.layers.conv2d(act3, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
#                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
#                                  name='conv4')
#         bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
#         act4 = myLeakyRelu(bn4)
       
#         # start from act4
#         dim = int(np.prod(act4.get_shape()[1:]))
#         fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')
      
        
#         w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
#                              initializer=tf.truncated_normal_initializer(stddev=0.02))
#         b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
#                              initializer=tf.constant_initializer(0.0))

#         # wgan just get rid of the sigmoid
#         logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
#         # dcgan
#         acted_out = tf.nn.sigmoid(logits)
#         return logits #, acted_out
def default_weights_initializer():
    return tf.contrib.layers.variance_scaling_initializer()

def discriminator(input, is_train, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        conv0 = convrelu2(name='conv0', inputs=input, filters=16, kernel_size=7, stride=2,activation=myLeakyRelu)
        conv0_b = tf.layers.batch_normalization(conv0)
        conv0_r =myLeakyRelu(conv0_b)


        conv1 = convrelu2(name='conv1', inputs=conv0_r, filters=32, kernel_size=5, stride=2,activation=myLeakyRelu)
        conv1_b = tf.layers.batch_normalization(conv1)
        conv1_r =myLeakyRelu(conv1_b)

        conv2 = convrelu2(name='conv2', inputs=conv1_r, filters=64, kernel_size=5, stride=2,activation=myLeakyRelu)
        conv2_b = tf.layers.batch_normalization(conv2)
        conv2_r =myLeakyRelu(conv2_b)

        conv3 = convrelu2(name='conv3', inputs=conv2_r, filters=128, kernel_size=5, stride=2,activation=myLeakyRelu)
        conv3_b = tf.layers.batch_normalization(conv3)
        conv3_r =myLeakyRelu(conv3_b)

        conv4 = convrelu2(name='conv4', inputs=conv3_r, filters=256, kernel_size=3, stride=2,activation=myLeakyRelu)
        conv4_b = tf.layers.batch_normalization(conv4)
        conv4_r =myLeakyRelu(conv4_b)


        conv5 = convrelu2(name='conv5', inputs=conv4_r, filters=512, kernel_size=3, stride=2,activation=myLeakyRelu)
        conv5_b = tf.layers.batch_normalization(conv5)
        conv5_r =myLeakyRelu(conv5_b)


        dense_slice_shape = conv5_r.get_shape().as_list()
        dense_slice_shape[-1] = 2
        units = 2

        # for i in range(1,len(dense_slice_shape)):
        #     units *= dense_slice_shape[i]

        dense5 = tf.layers.dense(
            tf.contrib.layers.flatten(tf.slice(conv5_r, [0,0,0,0], dense_slice_shape)),
            units=units,
            activation=None,
            kernel_initializer=default_weights_initializer(),
            name='dense5'
            )


        # dim = int(np.prod(conv3_r.get_shape()[1:]))
        # fc1 = tf.reshape(conv3_r, shape=[-1, dim], name='fc1')
      
        
        # w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
        #                      initializer=tf.truncated_normal_initializer(stddev=0.02))
        # b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
        #                      initializer=tf.constant_initializer(0.0))

        # # wgan just get rid of the sigmoid
        # logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')

        logits = tf.nn.sigmoid(dense5)

        # dcgan
        return logits, dense5 , conv4_r #, acted_out


def ggenerator(c,initializer):
    with tf.variable_scope('generator'):
        #Encoder

        enc0 = slim.conv2d(c,64,[3,3],padding="SAME",
            biases_initializer=None,activation_fn=tf.nn.leaky_relu,
            weights_initializer=initializer)
        enc0 = tf.space_to_depth(enc0,2)

        enc1 = slim.conv2d(enc0,128,[3,3],padding="SAME",
            activation_fn=myLeakyRelu,normalizer_fn=slim.batch_norm,
            weights_initializer=initializer)
        enc1 = tf.space_to_depth(enc1,2)

        enc2 = slim.conv2d(enc1,128,[3,3],padding="SAME",
            normalizer_fn=slim.batch_norm,activation_fn=myLeakyRelu,
            weights_initializer=initializer)
        enc2 = tf.space_to_depth(enc2,2)

        enc3 = slim.conv2d(enc2,256,[3,3],padding="SAME",
            normalizer_fn=slim.batch_norm,activation_fn=myLeakyRelu,
            weights_initializer=initializer)
        enc3 = tf.space_to_depth(enc3,2)
        
        #Decoder
        gen0 = slim.conv2d(
            enc3,num_outputs=256,kernel_size=[3,3],
            padding="SAME",normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.elu, weights_initializer=initializer)
        gen0 = tf.depth_to_space(gen0,2)

        gen1 = slim.conv2d(
            tf.concat([gen0,enc2],3),num_outputs=256,kernel_size=[3,3],
            padding="SAME",normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.elu,weights_initializer=initializer)
        gen1 = tf.depth_to_space(gen1,2)

        gen2 = slim.conv2d(
            tf.concat([gen1,enc1],3),num_outputs=128,kernel_size=[3,3],
            padding="SAME",normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.elu,weights_initializer=initializer)
        gen2 = tf.depth_to_space(gen2,2)

        gen3 = slim.conv2d(
            tf.concat([gen2,enc0],3),num_outputs=128,kernel_size=[3,3],
            padding="SAME",normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.elu, weights_initializer=initializer)
        gen3 = tf.depth_to_space(gen3,2)
        
        g_out = slim.conv2d(
            gen3,num_outputs=3,kernel_size=[1,1],padding="SAME",
            biases_initializer=None,activation_fn=tf.nn.tanh,
            weights_initializer=initializer)
        return 'a',g_out

def ddiscriminator(bottom,initializer, reuse=False):
    with tf.variable_scope('discriminator'):
        filters = [32,64,128,128]
        
        #Programatically define layers
        for i in range(len(filters)):
            if i == 0:
                layer = slim.conv2d(bottom,filters[i],[3,3],padding="SAME",scope='d'+str(i),
                    biases_initializer=None,activation_fn=myLeakyRelu,stride=[2,2],
                    reuse=reuse,weights_initializer=initializer)
            else:
                layer = slim.conv2d(bottom,filters[i],[3,3],padding="SAME",scope='d'+str(i),
                    normalizer_fn=slim.batch_norm,activation_fn=myLeakyRelu,stride=[2,2],
                    reuse=reuse,weights_initializer=initializer)
            bottom = layer

        dis_full = slim.fully_connected(slim.flatten(bottom),1024,activation_fn=myLeakyRelu,scope='dl',
            reuse=reuse, weights_initializer=initializer)

        d_out = slim.fully_connected(dis_full,1,activation_fn=None,scope='do',
            reuse=reuse, weights_initializer=initializer)

        logit = tf.nn.sigmoid(d_out)
        return logit, d_out, 'a'

