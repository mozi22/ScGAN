import os
import re
import time
import math
import logging
import network
import numpy as np
import losses_helper
import tensorflow as tf
from six.moves import xrange
import tensorflow.contrib.slim as slim
from tensorflow.python import debug as tf_debug
from tensorflow.python.client import device_lib
from datetime import datetime
# import ijremote

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']



# Training Variables

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('TRAIN_DIR', './ckpt/driving/kachra/',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_boolean('LOAD_FROM_CKPT', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_boolean('DEBUG_MODE', False,
                            """Run training in Debug Mode.""")

tf.app.flags.DEFINE_string('TOWER_NAME', 'tower',
                           """The name of the tower """)

tf.app.flags.DEFINE_integer('MAX_STEPS', 30000,
                            """Number of batches to run.""")


tf.app.flags.DEFINE_boolean('LOG_DEVICE_PLACEMENT', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer('EXAMPLES_PER_EPOCH_TRAIN', 200,
                            """How many samples are there in one epoch of training.""")

tf.app.flags.DEFINE_integer('EXAMPLES_PER_EPOCH_TEST', 100,
                            """How many samples are there in one epoch of testing.""")

tf.app.flags.DEFINE_integer('BATCH_SIZE', 1,
                            """How many samples are there in one epoch of testing.""")

tf.app.flags.DEFINE_integer('NUM_EPOCHS_PER_DECAY', 1,
                            """How many epochs per decay.""")

tf.app.flags.DEFINE_integer('SHUFFLE_BATCH_QUEUE_CAPACITY', 100,
                            """How many elements will be there in the queue to be dequeued.""")

tf.app.flags.DEFINE_integer('SHUFFLE_BATCH_THREADS', 48,
                            """How many elements will be there in the queue to be dequeued.""")

tf.app.flags.DEFINE_integer('SHUFFLE_BATCH_MIN_AFTER_DEQUEUE', 10,
                            """How many elements will be there in the queue to be dequeued.""")

tf.app.flags.DEFINE_integer('NUM_GPUS', len(get_available_gpus()),
                            """How many GPUs to use.""")

tf.app.flags.DEFINE_float('MOVING_AVERAGE_DECAY', 0.9999,
                            """How fast the learning rate should go down.""")

tf.app.flags.DEFINE_integer('TOTAL_TRAIN_EXAMPLES', 200,
                            """How many samples are there in one epoch of testing.""")


tf.app.flags.DEFINE_boolean('DISABLE_DISCRIMINATOR', False,
                            """Whether to log device placement.""")

# Testing Variables

tf.app.flags.DEFINE_integer('TOTAL_TEST_EXAMPLES', 100,
                            """How many samples are there in one epoch of testing.""")

tf.app.flags.DEFINE_integer('TEST_BATCH_SIZE', 16,
                            """How many samples are there in one epoch of testing.""")
 
# Polynomial Learning Rate
tf.app.flags.DEFINE_float('RMS_LEARNING_RATE', 2e-4,
                            """Where to start the learning.""")

tf.app.flags.DEFINE_float('G_START_LEARNING_RATE', 0.000099,
                            """Where to start the learning.""")
tf.app.flags.DEFINE_float('G_END_LEARNING_RATE', 0.000001,
                            """Where to end the learning.""")
tf.app.flags.DEFINE_float('G_POWER', 4,
                            """How fast the learning rate should go down.""")

tf.app.flags.DEFINE_float('D_START_LEARNING_RATE', 0.000099,
                            """Where to start the learning.""")
tf.app.flags.DEFINE_float('D_END_LEARNING_RATE', 0.000001,
                            """Where to end the learning.""")
tf.app.flags.DEFINE_float('D_POWER', 4,
                            """How fast the learning rate should go down.""")


tf.app.flags.DEFINE_float('D_GAUSSIAN_NOISE_ANNEALING_START', 0.2,
                            """Where to start the learning.""")
tf.app.flags.DEFINE_float('D_GAUSSIAN_NOISE_ANNEALING_END', 0,
                            """Where to end the learning.""")
tf.app.flags.DEFINE_float('D_POWER_ANNEALING', 2,
                            """How fast the learning rate should go down.""")

tf.app.flags.DEFINE_float('G_ITERATIONS', 5,
                            """How fast the learning rate should go down.""")

tf.app.flags.DEFINE_float('D_ITERATIONS', 5,
                            """How fast the learning rate should go down.""")




class DatasetReader:

    def __init__(self):
        # for testing
        self.X = tf.placeholder(dtype=tf.float32, shape=(FLAGS.TEST_BATCH_SIZE, 224, 384, 8))
        self.Y = tf.placeholder(dtype=tf.float32, shape=(FLAGS.TEST_BATCH_SIZE, 224, 384, 3))

        self.TRAIN_EPOCH = math.ceil(FLAGS.TOTAL_TRAIN_EXAMPLES / FLAGS.BATCH_SIZE)
        self.TEST_EPOCH = math.ceil(FLAGS.TOTAL_TEST_EXAMPLES / FLAGS.TEST_BATCH_SIZE)

        self.random_dim = [None, 100]
        self.random_input = tf.placeholder(tf.float32, shape= self.random_dim , name='rand_input')


    def train(self,features_train,features_test):

        self.global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # num_batches_per_epoch = (FLAGS.EXAMPLES_PER_EPOCH_TRAIN / FLAGS.BATCH_SIZE)
        # decay_steps = int(num_batches_per_epoch * FLAGS.NUM_EPOCHS_PER_DECAY)
        decay_steps = FLAGS.MAX_STEPS
        start_learning_rate = FLAGS.G_START_LEARNING_RATE
        end_learning_rate = FLAGS.G_END_LEARNING_RATE
        power = FLAGS.G_POWER

        learning_rate = tf.train.polynomial_decay(start_learning_rate, self.global_step,
                                                  decay_steps, end_learning_rate,
                                                  power=power)


        g_opt = tf.train.AdamOptimizer(learning_rate)



        decay_steps = FLAGS.MAX_STEPS
        start_learning_rate = FLAGS.D_START_LEARNING_RATE
        end_learning_rate = FLAGS.D_END_LEARNING_RATE
        power = FLAGS.D_POWER

        learning_rate = tf.train.polynomial_decay(start_learning_rate, self.global_step,
                                                  decay_steps, end_learning_rate,
                                                  power=power)

        d_opt = tf.train.AdamOptimizer(learning_rate)
    
        images, labels = tf.train.shuffle_batch(
                            [ features_train['input_n'] , features_train['label_n'] ],
                            batch_size=FLAGS.BATCH_SIZE,
                            capacity=FLAGS.SHUFFLE_BATCH_QUEUE_CAPACITY,
                            num_threads=FLAGS.SHUFFLE_BATCH_THREADS,
                            min_after_dequeue=FLAGS.SHUFFLE_BATCH_MIN_AFTER_DEQUEUE,
                            enqueue_many=False)

        # self.images_test, self.labels_test = tf.train.shuffle_batch(
        #                     [ features_test['input_n'] , features_test['label_n'] ],
        #                     batch_size=FLAGS.TEST_BATCH_SIZE,
        #                     capacity=FLAGS.SHUFFLE_BATCH_QUEUE_CAPACITY,
        #                     num_threads=FLAGS.SHUFFLE_BATCH_THREADS,
        #                     min_after_dequeue=FLAGS.SHUFFLE_BATCH_MIN_AFTER_DEQUEUE,
        #                     enqueue_many=False)
        
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [images, labels], capacity=FLAGS.SHUFFLE_BATCH_QUEUE_CAPACITY * FLAGS.NUM_GPUS)

        # self.batch_queue_test = tf.contrib.slim.prefetch_queue.prefetch_queue(
        #     [self.images_test, self.labels_test], capacity=FLAGS.SHUFFLE_BATCH_QUEUE_CAPACITY * FLAGS.NUM_GPUS)
        
        tower_grads_g = []
        tower_grads_d = []
        with tf.variable_scope(tf.get_variable_scope()):
          for i in xrange(FLAGS.NUM_GPUS):
            with tf.device('/gpu:%d' % i):
              with tf.name_scope('%s_%d' % ('tower', i)) as scope:

                # Dequeues one batch for the GPU
                image_batch, label_batch = batch_queue.dequeue()

                # Calculate the loss for one tower of the CIFAR model. This function
                # constructs the entire CIFAR model but shares the variables across
                # all towers.
                if FLAGS.DISABLE_DISCRIMINATOR == False:
                    self.loss_g, self.loss_d, g_var, d_var = self.tower_loss(scope, image_batch, label_batch)
                else:
                    self.loss_g, _, g_var, _ = self.tower_loss(scope, image_batch, label_batch)

                # clip discriminator weights
                # d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_var]


                # Reuse variables for the next tower.
                tf.get_variable_scope().reuse_variables()

                # Retain the summaries from the final tower.
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)


                # Calculate the gradients for the batch of data on this CIFAR tower.
                g_grads = g_opt.compute_gradients(self.loss_g,var_list=g_var)

                if FLAGS.DISABLE_DISCRIMINATOR == False:
                    d_grads = d_opt.compute_gradients(self.loss_d,var_list=d_var)


                # Keep track of the gradients across all towers.
                tower_grads_g.append(g_grads)

                if FLAGS.DISABLE_DISCRIMINATOR == False:
                    tower_grads_d.append(d_grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        g_grads = self.average_gradients(tower_grads_g)

        if FLAGS.DISABLE_DISCRIMINATOR == False:
            d_grads = self.average_gradients(tower_grads_d)

        # Add a summary to track the learning rate.
        # summaries.append(tf.summary.scalar('learning_rate', learning_rate))


        # Add histograms for gradients.
        for grad, var in g_grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/g_gradients', grad))

        if FLAGS.DISABLE_DISCRIMINATOR == False:
            for grad, var in d_grads:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.op.name + '/d_gradients', grad))


        # Apply the gradients to adjust the shared variables.
        apply_gradient_op_g = g_opt.apply_gradients(g_grads, global_step=self.global_step)

        if FLAGS.DISABLE_DISCRIMINATOR == False:
            apply_gradient_op_d = d_opt.apply_gradients(d_grads, global_step=self.global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.MOVING_AVERAGE_DECAY, self.global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op_g = tf.group(apply_gradient_op_g, variables_averages_op)

        if FLAGS.DISABLE_DISCRIMINATOR == False:
            train_op_d = tf.group(apply_gradient_op_d, variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        self.summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.

        tf_config = tf.ConfigProto(allow_soft_placement=True,
            log_device_placement=FLAGS.LOG_DEVICE_PLACEMENT)

        if 'dacky' in os.uname()[1]:
            logging.info('Dacky: Running with memory usage limits')
            # change tf_config for dacky to use only 1 GPU
            tf_config.gpu_options.per_process_gpu_memory_fraction = 0.6
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        else:
            # change tf_config for lmb_cluster so that GPU is visible and utilized
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        sess = tf.Session(config=tf_config)
        sess.run(init)


        if FLAGS.LOAD_FROM_CKPT == True:
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.TRAIN_DIR))


        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        # for debugging

        summary_writer = tf.summary.FileWriter(FLAGS.TRAIN_DIR, sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(FLAGS.TRAIN_DIR + '/test')


        # just to make sure we start from where we left, if load_from_ckpt = True
        loop_start = tf.train.global_step(sess, self.global_step)
        loop_stop = loop_start + FLAGS.MAX_STEPS

        if FLAGS.DEBUG_MODE:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        first_iteration = True

        # this will print in console for which time we are calculating the test loss.
        # first time or second time or third time and more
        test_loss_calculating_index = 1

        # main loop
        for step in range(loop_start,loop_stop):

            start_time = time.time()



            duration = time.time() - start_time
            

            if step % 10 == 0 or first_iteration==True:
                num_examples_per_step = FLAGS.BATCH_SIZE * FLAGS.NUM_GPUS
                examples_per_sec = num_examples_per_step / (duration + 1e-5) 
                sec_per_batch = duration / FLAGS.NUM_GPUS
                first_iteration = False

            if FLAGS.DISABLE_DISCRIMINATOR == False:
            # discriminator
                # self.log()
                for k in range(FLAGS.D_ITERATIONS):
                    _, loss_value_d = sess.run([train_op_d, self.loss_d])
        
                    assert not np.isnan(loss_value_d), 'Discriminator Model diverged with loss = NaN'

                    format_str = ('loss = %.15f (%.1f examples/sec; %.3f sec/batch, %02d Step, Discriminator)')
                    self.log(message=(format_str % (np.log10(loss_value_d),examples_per_sec, sec_per_batch,step)))



            # generator
            # self.log()
            for k in range(FLAGS.G_ITERATIONS):
                _, loss_value_g = sess.run([train_op_g, self.loss_g])
    
                assert not np.isnan(loss_value_g), 'Generator Model  diverged with loss = NaN'

                format_str = ('loss = %.15f (%.1f examples/sec; %.3f sec/batch, %02d Step, Generator)')
                self.log(message=(format_str % (np.log10(loss_value_g),examples_per_sec, sec_per_batch, step)))

            if step % 100 == 0:
                summary_str = sess.run(self.summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.MAX_STEPS:
                checkpoint_path = os.path.join(FLAGS.TRAIN_DIR, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


             # # after every 10 epochs. calculate test loss
            if step % (self.TRAIN_EPOCH * 10) == 0 and first_iteration==True:

                message = 'Printing Test loss for '+str(test_loss_calculating_index)+' time'

                self.log()
                self.log(message)
                self.log()

                self.perform_testing(sess,step)

                # increment index to know how many times we've calculated the test loss
                test_loss_calculating_index = test_loss_calculating_index + 1


            # if step == 4000:
            #     break

        summary_writer.close()

    def tower_loss(self,scope, images, labels):
        """Calculate the total loss on a single tower running the CIFAR model.
        Args:
        scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
        images: Images. 4D tensor of shape [batch_size, height, width, 3].
        labels: Labels. 1D tensor of shape [batch_size].
        Returns:
         Tensor of shape [] containing the total loss for a batch of data

        """
        g_total_loss, d_total_loss, g_vars, d_vars = -1, -1, -1 ,-1


        network_input_images, network_input_labels = self.get_network_input_forward(images,labels)
        # network_input_labels = tf.image.resize_images(network_input_labels,[128,128],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # network_input_labels_u = network_input_labels[:,:,:,0] * 0.571428571
        # network_input_labels_v = network_input_labels[:,:,:,1] * 0.333333333
        # network_input_labels_w = network_input_labels[:,:,:,2]

        # network_input_labels_u = tf.expand_dims(network_input_labels_u,axis=-1)
        # network_input_labels_v = tf.expand_dims(network_input_labels_v,axis=-1)
        # network_input_labels_w = tf.expand_dims(network_input_labels_w,axis=-1)

        # network_input_labels = tf.concat([network_input_labels_u,network_input_labels_v,network_input_labels_w],axis=3)
        network_input_labels = network_input_labels[:,:,:,0:2]
        # network_input_images_back, network_input_labels_back = self.get_network_input_backward(images,labels)
        # FB = forward-backward
        # concatenated_FB_images = tf.concat([network_input_images,network_input_images_back],axis=0)

        # backward_flow_images = losses_helper.forward_backward_loss()
        dim = [FLAGS.BATCH_SIZE,100]
        noise = tf.random_uniform(dim)


        noise_annealer = tf.train.polynomial_decay(FLAGS.D_GAUSSIAN_NOISE_ANNEALING_START, self.global_step,
                                                  FLAGS.MAX_STEPS, FLAGS.D_GAUSSIAN_NOISE_ANNEALING_END,
                                                  power=FLAGS.D_POWER_ANNEALING)


        tf.summary.scalar('noise annealer',noise_annealer)
        disc_noise = tf.random_normal(network_input_labels.get_shape(),0,noise_annealer)

        real_flow = network_input_labels

        # adding gaussian noise to discriminator.
        # real_flow = real_flow + disc_noise

        fake_flow = network.generator(noise, dim[1], True,False,network_input_images)

        concated_flows_u = tf.concat([network_input_labels[:,:,:,0:1],fake_flow[:,:,:,0:1]],axis=-2)
        concated_flows_v = tf.concat([network_input_labels[:,:,:,1:2],fake_flow[:,:,:,1:2]],axis=-2)


        tf.summary.image('real_fake_flow_u',concated_flows_u)
        tf.summary.image('real_fake_flow_v',concated_flows_v)

        if FLAGS.DISABLE_DISCRIMINATOR == False:
            real_flow_d, real_flow_logits_d  = network.discriminator(real_flow,True)
            fake_flow_d, fake_flow_logits_d = network.discriminator(fake_flow,True, reuse=True)

            # d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)  # This optimizes the discriminator.
            # g_loss = -tf.reduce_mean(fake_result)  # This optimizes the generator.

            # discriminator loss

            d_loss_1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_flow_logits_d,labels=tf.ones_like(real_flow_d))
            d_loss_2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_flow_logits_d,labels=tf.zeros_like(fake_flow_d))

            d_loss_1 = tf.reduce_mean(d_loss_1)
            d_loss_2 = tf.reduce_mean(d_loss_2)
            d_total_loss =  d_loss_1 + d_loss_2

            tf.summary.scalar('d_loss_real',d_loss_1)
            tf.summary.scalar('d_loss_fake',d_loss_2)



        # generator loss
        lambda_adversarial = 0.01
        # here we'll try to just minimize the epe loss between fake_image and the original flow values labels.
        g_epe_loss = losses_helper.endpoint_loss(network_input_labels,fake_flow)

        # here we are passing G(z) -> fake_result after passing in random distribution
        # and passing the labels as 1.
        # in short we're saying this is the real image not the fake one.

        if FLAGS.DISABLE_DISCRIMINATOR == False:
            # g_adversarial_loss_labeled = lambda_adversarial * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fake_flow_logits_d,labels=tf.ones_like(fake_flow_d)))
            g_adversarial_loss_labeled = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_flow_logits_d,labels=tf.ones_like(fake_flow_d)))
            # g_total_loss = g_adversarial_loss_labeled
            g_total_loss = g_adversarial_loss_labeled + g_epe_loss
            d_total_loss = tf.losses.compute_weighted_loss(d_total_loss)
            # d_total_loss = tf.losses.compute_weighted_loss(d_loss_1)
            # d_total_loss = tf.losses.compute_weighted_loss(d_loss_2)
            tf.summary.scalar('total_discrimnator_loss',d_total_loss)
        else:
            g_total_loss = g_epe_loss

        g_total_loss = tf.losses.compute_weighted_loss(g_total_loss)


        tf.summary.scalar('generator_endpoint_loss',g_epe_loss)
        tf.summary.scalar('total_generator_loss',g_total_loss)



        t_vars = tf.trainable_variables()

        if FLAGS.DISABLE_DISCRIMINATOR == False:
            d_vars = [var for var in t_vars if 'dis' in var.name]


        g_vars = [var for var in t_vars if 'gen' in var.name]


        return g_total_loss, d_total_loss, g_vars, d_vars


    def get_network_input_forward(self,image_batch,label_batch):
        return image_batch[:,0,:,:,:], label_batch[:,0,:,:,:]

    def get_network_input_backward(self,image_batch,label_batch):
        return image_batch[:,1,:,:,:], label_batch[:,1,:,:,:]

    def average_gradients(self,tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
            tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:

                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads


    def perform_testing(self,sess,step):
    

        for step in range(step,step + self.TEST_EPOCH):

            image_batch, label_batch = self.batch_queue_test.dequeue()
            image_batch, label_batch = self.get_network_input(image_batch,label_batch)

            image,label = sess.run([image_batch, label_batch])

            loss_value,summary_str = sess.run([self.loss,self.summary_op],feed_dict={self.X: image, self.Y: label})

            self.test_summary_writer.add_summary(summary_str, step)


            format_str = ('%s: step %d, loss = %.15f')
            self.log(message=(format_str % (datetime.now(), step, np.log10(loss_value))))


        self.log()
        self.log(message='Continue Training ...')
        self.log()


    def log(self,message=' '):
        print(message)
