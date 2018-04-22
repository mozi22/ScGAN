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
        activation=myLeakyRelu,
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

    tmp = convrelu2(
        inputs=inp,
        filters=24,
        kernel_size=3,
        stride=1,
        name="conv1_pred_flow"
    )
    
    output = convrelu2(
        inputs=tmp,
        filters=2,
        kernel_size=3,
        stride=1,
        name="conv2_pred_flow"
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

def generator(input, random_dim, is_train, reuse=False):
    c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32 # channel num
    s4 = 4
    output_dim = CHANNEL  # RGB image
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable('w1', shape=[random_dim, s4 * s4 * c4], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape=[c4 * s4 * s4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')
         #Convolution, bias, activation, repeat! 
        conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn1')
        act1 = tf.nn.relu(bn1, name='act1')
        # 8*8*256
        #Convolution, bias, activation, repeat! 
        conv2 = tf.layers.conv2d_transpose(act1, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = tf.nn.relu(bn2, name='act2')
        # 16*16*128
        conv3 = tf.layers.conv2d_transpose(act2, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = tf.nn.relu(bn3, name='act3')
        # 32*32*64
        conv4 = tf.layers.conv2d_transpose(act3, c32, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = tf.nn.relu(bn4, name='act4')
        # 64*64*32
        conv5 = tf.layers.conv2d_transpose(act4, c64, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv5')
        bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn5')
        act5 = tf.nn.relu(bn5, name='act5')
        
        #128*128*3
        conv6 = tf.layers.conv2d_transpose(act5, output_dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv6')
        # bn6 = tf.contrib.layers.batch_norm(conv6, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn6')
        act6 = tf.nn.tanh(conv6, name='act6')
        return act6


def discriminator(input, is_train, reuse=False):
    c2, c4, c8, c16 = 64, 128, 256, 512  # channel num: 64, 128, 256, 512
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()

        #Convolution, activation, bias, repeat! 
        conv1 = tf.layers.conv2d(input, c2, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')
        act1 = lrelu(conv1, n='act1')
         #Convolution, activation, bias, repeat! 
        conv2 = tf.layers.conv2d(act1, c4, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = lrelu(bn2, n='act2')
        #Convolution, activation, bias, repeat! 
        conv3 = tf.layers.conv2d(act2, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = lrelu(bn3, n='act3')
         #Convolution, activation, bias, repeat! 
        conv4 = tf.layers.conv2d(act3, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = lrelu(bn4, n='act4')
       
        # start from act4
        dim = int(np.prod(act4.get_shape()[1:]))
        fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')
      
        
        w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

        # wgan just get rid of the sigmoid
        logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
        # dcgan
        acted_out = tf.nn.sigmoid(logits)
        return logits #, acted_out
