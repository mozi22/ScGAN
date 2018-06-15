import tensorflow as tf
import numpy as np
import lmbspecialops as sops
from PIL import Image
import ijremote as ij
def tf_record_input_pipeline(filenames,version='1'):

    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer(filenames,name='InputProducerV'+version,shuffle=False)

    # Define a reader and read the next record
    recordReader = tf.TFRecordReader(name="TfReaderV"+version)

    key, fullExample = recordReader.read(filename_queue)

    # Decode the record read by the reader
    features = tf.parse_single_example(fullExample, {
        'depth1': tf.FixedLenFeature([], tf.string),
        'depth2': tf.FixedLenFeature([], tf.string),
        'depth_change': tf.FixedLenFeature([], tf.string),
        'opt_flow': tf.FixedLenFeature([], tf.string),
        'image1': tf.FixedLenFeature([], tf.string),
        'image2': tf.FixedLenFeature([], tf.string)
    },
    name="ExampleParserV"+version)

    # Convert the image data from binary back to arrays(Tensors)
    depth1 = tf.decode_raw(features['depth1'], tf.float32)
    depth2 = tf.decode_raw(features['depth2'], tf.float32)
    depth_chng = tf.decode_raw(features['depth_change'], tf.float32)

    image1 = tf.decode_raw(features['image1'], tf.uint8)
    image2 = tf.decode_raw(features['image2'], tf.uint8)
    opt_flow = tf.decode_raw(features['opt_flow'], tf.float32)

    input_pipeline_dimensions = [224, 384]
    image1 = tf.to_float(image1)
    image2 = tf.to_float(image2)


    # reshape data to its original form
    image1 = tf.reshape(image1, [input_pipeline_dimensions[0],input_pipeline_dimensions[1], 3],name="reshape_img1")
    image2 = tf.reshape(image2, [input_pipeline_dimensions[0],input_pipeline_dimensions[1], 3],name="reshape_img2")

    depth1 = tf.reshape(depth1, [input_pipeline_dimensions[0],input_pipeline_dimensions[1]],name="reshape_disp1")
    depth2 = tf.reshape(depth2, [input_pipeline_dimensions[0],input_pipeline_dimensions[1]],name="reshape_disp2")

    optical_flow = tf.reshape(opt_flow, [input_pipeline_dimensions[0],input_pipeline_dimensions[1],2],name="reshape_opt_flow")
    depth_chng = tf.reshape(depth_chng,[input_pipeline_dimensions[0],input_pipeline_dimensions[1]],name="reshape_depth_change")


    image1 = tf.divide(image1,[255])
    image2 = tf.divide(image2,[255])

    # return train_for_opticalflow(image1,image2,optical_flow)
    return train_for_sceneflow(image1,image2,depth1,depth2,depth_chng,optical_flow)

def _parse_function(example_proto):

    features = tf.parse_single_example(example_proto, {
        'depth1': tf.FixedLenFeature([], tf.string),
        'depth2': tf.FixedLenFeature([], tf.string),
        'depth_change': tf.FixedLenFeature([], tf.string),
        'opt_flow': tf.FixedLenFeature([], tf.string),
        'image1': tf.FixedLenFeature([], tf.string),
        'image2': tf.FixedLenFeature([], tf.string)
    },
    name="ExampleParserV")
    # Convert the image data from binary back to arrays(Tensors)
    depth1 = tf.decode_raw(features['depth1'], tf.float32)
    depth2 = tf.decode_raw(features['depth2'], tf.float32)
    depth_chng = tf.decode_raw(features['depth_change'], tf.float32)

    image1 = tf.decode_raw(features['image1'], tf.uint8)
    image2 = tf.decode_raw(features['image2'], tf.uint8)
    opt_flow = tf.decode_raw(features['opt_flow'], tf.float32)

    input_pipeline_dimensions = [224, 384]
    image1 = tf.to_float(image1)
    image2 = tf.to_float(image2)


    # reshape data to its original form
    image1 = tf.reshape(image1, [input_pipeline_dimensions[0],input_pipeline_dimensions[1], 3],name="reshape_img1")
    image2 = tf.reshape(image2, [input_pipeline_dimensions[0],input_pipeline_dimensions[1], 3],name="reshape_img2")

    depth1 = tf.reshape(depth1, [input_pipeline_dimensions[0],input_pipeline_dimensions[1]],name="reshape_disp1")
    depth2 = tf.reshape(depth2, [input_pipeline_dimensions[0],input_pipeline_dimensions[1]],name="reshape_disp2")

    optical_flow = tf.reshape(opt_flow, [input_pipeline_dimensions[0],input_pipeline_dimensions[1],2],name="reshape_opt_flow")
    depth_chng = tf.reshape(depth_chng,[input_pipeline_dimensions[0],input_pipeline_dimensions[1]],name="reshape_depth_change")


    image1 = tf.divide(image1,[255])
    image2 = tf.divide(image2,[255])

    final_result = train_for_sceneflow(image1,image2,depth1,depth2,depth_chng,optical_flow)

    return final_result["input_n"], final_result["label_n"]

def _parse_function_ptb(example_proto):

    features = tf.parse_single_example(example_proto, {
        'depth1': tf.FixedLenFeature([], tf.string),
        'depth2': tf.FixedLenFeature([], tf.string),
        'image1': tf.FixedLenFeature([], tf.string),
        'image2': tf.FixedLenFeature([], tf.string)
    },
    name="ExampleParserV")

    # Convert the image data from binary back to arrays(Tensors)
    depth1 = tf.decode_raw(features['depth1'], tf.float32)
    depth2 = tf.decode_raw(features['depth2'], tf.float32)

    image1 = tf.decode_raw(features['image1'], tf.uint8)
    image2 = tf.decode_raw(features['image2'], tf.uint8)

    input_pipeline_dimensions = [224, 384]
    image1 = tf.to_float(image1)
    image2 = tf.to_float(image2)


    # reshape data to its original form
    image1 = tf.reshape(image1, [input_pipeline_dimensions[0],input_pipeline_dimensions[1], 3],name="reshape_img1")
    image2 = tf.reshape(image2, [input_pipeline_dimensions[0],input_pipeline_dimensions[1], 3],name="reshape_img2")

    depth1 = tf.reshape(depth1, [input_pipeline_dimensions[0],input_pipeline_dimensions[1]],name="reshape_disp1")
    depth2 = tf.reshape(depth2, [input_pipeline_dimensions[0],input_pipeline_dimensions[1]],name="reshape_disp2")

    optical_flow1 = tf.zeros_like(depth1)
    optical_flow2 = tf.zeros_like(depth1)
    depth_chng = tf.zeros_like(depth1)

    optical_flow1 = tf.expand_dims(optical_flow1,axis=2)
    optical_flow2 = tf.expand_dims(optical_flow2,axis=2)
    optical_flow = tf.concat([optical_flow1,optical_flow2],axis=2)

    image1 = tf.divide(image1,[255])
    image2 = tf.divide(image2,[255])

    final_result = train_for_sceneflow(image1,image2,depth1,depth2,depth_chng,optical_flow)

    return final_result["input_n"], final_result["label_n"]



def read_with_dataset_api(batch_size,filenames,version='1'):

    # parallel cpu calls
    num_parallel_calls = 16
    buffer_size = 50

    mapped_data = []

    for name in filenames:
        data = tf.data.TFRecordDataset(name)

        if 'ptb' in name:
            data = data.map(map_func=_parse_function_ptb, num_parallel_calls=num_parallel_calls)
        else:
            data = data.map(map_func=_parse_function, num_parallel_calls=num_parallel_calls)
        mapped_data.append(data)

    data = tuple(mapped_data)

    dataset = tf.data.Dataset.zip(data)

    dataset = dataset.shuffle(buffer_size=50).repeat().apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.prefetch(batch_size)

    # testing_ds_api(dataset)

    return dataset



def testing_ds_api(dataset):

    dataset = dataset.make_initializable_iterator()
    sess = tf.InteractiveSession()
    sess.run(dataset.initializer)

    next_batch = dataset.get_next()

    final_img_batch, final_lbl_batch = combine_batches_from_datasets(next_batch)

    next_batch_forward = final_img_batch[:,0,:,:,:]
    next_batch_backward = final_img_batch[:,1,:,:,:]

    forward, backward = sess.run([next_batch_forward,next_batch_backward])
    ij.setHost('tcp://linus:13463')
    ij.setImage('myimage_f',np.transpose(forward,[0,3,1,2]))
    ij.setImage('myimage_b',np.transpose(backward,[0,3,1,2]))


def combine_batches_from_datasets(batches):

    imgs = []
    lbls = []

    # driving

    # batches[x][y] = (4, 2, 224, 384, 8)

    imgs.append(batches[0][0])
    lbls.append(batches[0][1])

    imgs.append(batches[1][0])
    lbls.append(batches[1][1])

    imgs.append(batches[2][0])
    lbls.append(batches[2][1])


    final_img_batch = tf.concat(tuple(imgs),axis=0)
    final_lbl_batch = tf.concat(tuple(lbls),axis=0)

    return final_img_batch, final_lbl_batch


def train_for_opticalflow(image1,image2,optical_flow):

    img_pair_rgb = tf.concat([image1,image2],axis=-1)
    img_pair_rgb_swapped = tf.concat([image2,image1],axis=-1)

    optical_flow_swapped = tf.zeros(optical_flow.get_shape())

    # inputt = divide_inputs_to_patches(img_pair,8)
    # label = divide_inputs_to_patches(label_pair,3)

    # padding_input = tf.constant([[0, 0],[5, 4],[0, 0]])
    x_dimension_padding = tf.constant([[4, 4],[0, 0],[0,0]])
    # padding2 = tf.constant([[4, 4],[0,0]])

    padded_img_pair_rgb = tf.pad(img_pair_rgb,x_dimension_padding,'CONSTANT')
    padded_optical_flow = tf.pad(optical_flow,x_dimension_padding,'CONSTANT')

    padded_img_pair_rgb_swapped = tf.pad(img_pair_rgb_swapped,x_dimension_padding,'CONSTANT')
    padded_optical_flow_swapped = tf.pad(optical_flow_swapped,x_dimension_padding,'CONSTANT')

    fb_rgb_img_pair = tf.stack([padded_img_pair_rgb,padded_img_pair_rgb_swapped])
    fb_rgb_optflow = tf.stack([padded_optical_flow,padded_optical_flow_swapped])

    return {
        'input_n': fb_rgb_img_pair,
        'label_n': fb_rgb_optflow
    }

def decode_pfm(path):
    data = loadpfm.readPFM(path)
    return data[0]

def decode_webp(path):
    data = Image.open(path)
    return np.array(data)


def train_for_sceneflow(image1,image2,depth1,depth2,depth_chng,optical_flow):

    max_depth1 = tf.reduce_max(depth1)
    depth1 = depth1 / max_depth1
    depth2 = depth2 / max_depth1

    depth1 = sops.replace_nonfinite(depth1)
    depth2 = sops.replace_nonfinite(depth2)

    image1 = combine_depth_values(image1,depth1,2)
    image2 = combine_depth_values(image2,depth2,2)

    img_pair_rgbd = tf.concat([image1,image2],axis=-1)
    img_pair_rgbd_swapped = tf.concat([image2,image1],axis=-1)

    # optical_flow = optical_flow / 50
    # comment for optical flow. Uncomment for Sceneflow
    optical_flow_with_depth_change = combine_depth_values(optical_flow,depth_chng,2)
    optical_flow_with_depth_change_swapped = tf.zeros(optical_flow_with_depth_change.get_shape())

    # inputt = divide_inputs_to_patches(img_pair,8)
    # label = divide_inputs_to_patches(label_pair,3)

    # padding_input = tf.constant([[0, 0],[5, 4],[0, 0]])
    # x_dimension_padding = tf.constant([[4, 4],[0, 0],[0,0]])
    # padding2 = tf.constant([[4, 4],[0,0]])
    # padded_img_pair_rgbd = tf.pad(img_pair_rgbd,x_dimension_padding,'CONSTANT')
    # padded_optical_flow_with_depth_change = tf.pad(optical_flow_with_depth_change,x_dimension_padding,'CONSTANT')

    # padded_img_pair_rgbd_swapped = tf.pad(img_pair_rgbd_swapped,x_dimension_padding,'CONSTANT')
    # padded_optical_flow_with_depth_change_swapped = tf.pad(optical_flow_with_depth_change_swapped,x_dimension_padding,'CONSTANT')

    fb_rgbd_img_pair = tf.stack([img_pair_rgbd,img_pair_rgbd_swapped])
    fb_rgbd_optflow_with_depth_change = tf.stack([optical_flow_with_depth_change,optical_flow_with_depth_change_swapped])

    return {
        'input_n': fb_rgbd_img_pair,
        'label_n': fb_rgbd_optflow_with_depth_change
    }

def test(img_pair,img_pair2):

    sess = tf.InteractiveSession()
    # img_1 = tf.constant([[[1,2],[5,6]],[[9,10],[13,14]]],dtype=tf.float32)
    # img_2 = tf.constant([[[3,4],[7,8]],[[11,12],[15,16]]],dtype=tf.float32)
    # img_3 = tf.constant([[[1,2],[5,6]],[[9,10],[13,14]]],dtype=tf.float32)
    # img_4 = tf.constant([[[3,4],[7,8]],[[11,12],[15,16]]],dtype=tf.float32)

    # img_pair = tf.concat([img_1,img_2],axis=-1)
    # img_pair2 = tf.concat([img_2,img_1],axis=-1)


    # img_pair_final2 = tf.constant([[[3,4,1,2],[7,8,5,6]],[[11,12,9,10],[15,16,13,14]]],dtype=tf.float32)

    img_pair_final = tf.stack([img_pair,img_pair2])

    labels_final = tf.constant([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]],dtype=tf.float32)
    labels_final = tf.zeros(labels_final.get_shape(),dtype=tf.float32)

    images, labels = tf.train.shuffle_batch(
                        [ img_pair_final , labels_final ],
                        batch_size=16,
                        capacity=100,
                        num_threads=48,
                        min_after_dequeue=1,
                        enqueue_many=False)

    images = tf.Print(images,[images[0,0,:,:,0:3]],summarize=10000,message='jalalala')
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    # print(sess.run(images[0,0,:,:,0:3]))
    coord.request_stop()
    coord.join(threads)
# combines the depth value in the image RGB values to make it an RGBD tensor.
# where the resulting tensor will have depth values in the 4th element of 3rd dimension i.e [0][0][3].
# where [x][x][0] = R, [x][x][1] = G, [x][x][2] = B, [x][x][3] = D
def combine_depth_values(image,depth,rank):
    depth = tf.expand_dims(depth,rank)
    return tf.concat([image,depth],rank)


def warp(img,flow,input_pipeline_dimensions):
    x = list(range(0,input_pipeline_dimensions[1]))
    y = list(range(0,input_pipeline_dimensions[0]))
    X, Y = tf.meshgrid(x, y)

    X = tf.cast(X,np.float32) + flow[:,:,0]
    Y = tf.cast(Y,np.float32) + flow[:,:,1]

    con = tf.stack([X,Y])
    result = tf.transpose(con,[1,2,0])
    result = tf.expand_dims(result,0)


    return tf.contrib.resampler.resampler(img[np.newaxis,:,:,:],result)


def divide_inputs_to_patches(image,last_dimension):

    
    image = tf.expand_dims(image,0)
    ksize = [1, 54, 96, 1]

    image_patches = tf.extract_image_patches(
        image, ksize, ksize, [1, 1, 1, 1], 'VALID')
    image_patches_reshaped = tf.reshape(image_patches, [-1, 54, 96, last_dimension])

    return image_patches_reshaped



def get_depth_chng_from_disparity_chng(disparity,disparity_change):

    disparity_change = tf.add(disparity,disparity_change)

    depth1 = get_depth_from_disparity(disparity)
    calcdepth = get_depth_from_disparity(disparity_change)

    return tf.subtract(depth1,calcdepth)


def get_depth_from_disparity(disparity):
    # focal_length = 35
    # baseline = 1
    # focal_length * baseline = 35

    focal_length = tf.constant([35],dtype=tf.float32)

    disp_to_depth = tf.divide(focal_length,disparity)

    return disp_to_depth
