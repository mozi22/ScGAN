import network
import tensorflow as tf
import losses_helper as lhpl
import helpers as hpl
from PIL import Image
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('PARENT_FOLDER', '../dataset_synthetic/driving/',
                           """The root folder for the dataset """)

tf.app.flags.DEFINE_string('PARENT_FOLDER_PTB', '../dataset_ptb/ValidationSet/bear_front/',
                           """The root folder for the dataset """)

tf.app.flags.DEFINE_string('IMG1',  'frames_finalpass_webp/35mm_focallength/scene_backwards/fast/left/',
                           """The name of the tower """)

tf.app.flags.DEFINE_string('IMG2',  'frames_finalpass_webp/35mm_focallength/scene_backwards/fast/left/',
                           """The name of the tower """)

tf.app.flags.DEFINE_string('DISPARITY1', 'disparity/35mm_focallength/scene_backwards/fast/left/',
                           """The name of the tower """)

tf.app.flags.DEFINE_string('DISPARITY2', 'disparity/35mm_focallength/scene_backwards/fast/left/',
                           """The name of the tower """)

tf.app.flags.DEFINE_string('FLOW', 'optical_flow/35mm_focallength/scene_backwards/fast/into_future/left/',
                           """The name of the tower """)

tf.app.flags.DEFINE_string('DISPARITY_CHNG', 'disparity_change/35mm_focallength/scene_backwards/fast/into_future/left/',
                           """The name of the tower """)

tf.app.flags.DEFINE_string('CKPT_FOLDER', 'ckpt/driving/epe/',
                           """The name of the tower """)

IMG1_NUMBER = '0001'
IMG2_NUMBER = '0002'


FLAGS.IMG1 = FLAGS.PARENT_FOLDER + FLAGS.IMG1 + IMG1_NUMBER + '.webp'
FLAGS.IMG2 = FLAGS.PARENT_FOLDER + FLAGS.IMG2 + IMG2_NUMBER + '.webp'
FLAGS.DISPARITY1 = FLAGS.PARENT_FOLDER + FLAGS.DISPARITY1 + IMG1_NUMBER + '.pfm'
FLAGS.DISPARITY2 = FLAGS.PARENT_FOLDER + FLAGS.DISPARITY2 + IMG2_NUMBER + '.pfm'
FLAGS.DISPARITY_CHNG = FLAGS.PARENT_FOLDER + FLAGS.DISPARITY_CHNG + IMG1_NUMBER + '.pfm'
FLAGS.FLOW = FLAGS.PARENT_FOLDER + FLAGS.FLOW + 'OpticalFlowIntoFuture_' + IMG1_NUMBER + '_L.pfm'

def get_depth_from_disp(disparity):
	focal_length = 1050.0
	disp_to_depth = disparity / focal_length
	return disp_to_depth

def combine_depth_values(img,depth):
	depth = np.expand_dims(depth,2)
	return np.concatenate((img,depth),axis=2)

def denormalize_flow(flow):

	u = flow[:,:,:,0] * input_size[0]
	v = flow[:,:,:,1] * input_size[1]
	# w = flow[:,:,2] * self.max_depth_driving_chng
	
	flow = np.stack((u,v),axis=3)
	
	return flow

def parse_input(img1,img2,disp1,disp2):
	img1 = Image.open(img1)
	img2 = Image.open(img2)

	disp1 = hpl.readPFM(disp1)[0]
	disp2 = hpl.readPFM(disp2)[0]

	disp1 = Image.fromarray(disp1,mode='F')
	disp2 = Image.fromarray(disp2,mode='F')


	img1 = img1.resize(input_size, Image.BILINEAR)
	img2 = img2.resize(input_size, Image.BILINEAR)

	disp1 = disp1.resize(input_size, Image.NEAREST)
	disp2 = disp2.resize(input_size, Image.NEAREST)

	disp1 = np.array(disp1,dtype=np.float32)
	disp2 = np.array(disp2,dtype=np.float32)

	depth1 = get_depth_from_disp(disp1)
	depth2 = get_depth_from_disp(disp2)

	# normalize
	depth1 = depth1 / np.max(depth1)
	depth2 = depth2 / np.max(depth1)

	img1_orig = np.array(img1)
	img2_orig = np.array(img2)

	img1 = img1_orig / 255
	img2 = img1_orig / 255

	rgbd1 = combine_depth_values(img1,depth1)
	rgbd2 = combine_depth_values(img2,depth2)

	img_pair = np.concatenate((rgbd1,rgbd2),axis=2)

				# optical_flow
	return img_pair, img2_orig

def downsample_opt_flow(data,size):

	u = data[:,:,0]
	v = data[:,:,1]
	
	dt = Image.fromarray(u)
	dt = dt.resize(size, Image.NEAREST)

	dt2 = Image.fromarray(v)
	dt2 = dt2.resize(size, Image.NEAREST)
	u = np.array(dt)
	v = np.array(dt2)

	return np.stack((u,v),axis=2)

def read_gt(opt_flow,input_size):
	opt_flow = hpl.readPFM(opt_flow)[0]
	opt_flow = downsample_opt_flow(opt_flow,input_size)
	return opt_flow

def predict(img_pair,optical_flow):

	optical_flow = downsample_opt_flow(optical_flow,(192,112))

	img_pair = np.expand_dims(img_pair,axis=0)
	optical_flow = np.expand_dims(optical_flow,axis=0)

	feed_dict = {
		X: img_pair,
		Y: optical_flow
	}

	loss, v = sess.run([loss_result,predict_flow2],feed_dict=feed_dict)

	return denormalize_flow(v), loss

def normalizeOptFlow(flow,input_size):

	# remove the values bigger than the image size
	flow[:,:,0][flow[:,:,0] > input_size[0] ] = 0
	flow[:,:,1][flow[:,:,1] > input_size[1] ] = 0

	# separate the u and v values 
	flow_u = flow[:,:,0]
	flow_v = flow[:,:,1]

	# normalize the values by the image dimensions
	flow_u = flow_u / input_size[0]
	flow_v = flow_v / input_size[1]



	# combine them back and return
	return np.dstack((flow_u,flow_v))

def further_resize_imgs(network_input_images):
    network_input_images = tf.image.resize_images(network_input_images,[112,192],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return network_input_images

def further_resize_lbls(network_input_labels):

	network_input_labels = tf.image.resize_images(network_input_labels,[112,192],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

	network_input_labels_u = network_input_labels[:,:,:,0] * 0.5
	network_input_labels_v = network_input_labels[:,:,:,1] * 0.5

	network_input_labels_u = tf.expand_dims(network_input_labels_u,axis=-1)
	network_input_labels_v = tf.expand_dims(network_input_labels_v,axis=-1)

	network_input_labels = tf.concat([network_input_labels_u,network_input_labels_v],axis=3)

	return network_input_labels

def perform_testing():

	
	optical_flow = read_gt(FLAGS.FLOW,input_size)

	img_pair, img2_orig = parse_input(FLAGS.IMG1,FLAGS.IMG2,FLAGS.DISPARITY1,FLAGS.DISPARITY2)
	optical_flow = normalizeOptFlow(optical_flow,(192,112))

	predicted_flow, loss = predict(img_pair,optical_flow)

	img2_to_tensor = tf.expand_dims(tf.convert_to_tensor(img2_orig,dtype=tf.float32),axis=0)
	pred_flow_to_tensor = tf.convert_to_tensor(predicted_flow,dtype=tf.float32)
	orig_flow_to_tensor = tf.expand_dims(tf.convert_to_tensor(optical_flow,dtype=tf.float32),axis=0)


	img2_to_tensor = further_resize_imgs(img2_to_tensor)
	orig_flow_to_tensor = further_resize_lbls(orig_flow_to_tensor)

	warped_img =  lhpl.flow_warp(img2_to_tensor,pred_flow_to_tensor)

	warped_img = sess.run(warped_img)
	warped_img = np.squeeze(warped_img)

	# Image.fromarray(np.uint8(img2_orig)).show()
	Image.fromarray(np.uint8(warped_img)).show()
	print(loss)

def load_model_ckpt(sess,filename):
	saver = tf.train.Saver()
	saver.restore(sess, tf.train.latest_checkpoint(filename))

input_size = (384,224)
sess = tf.InteractiveSession()
X = tf.placeholder(dtype=tf.float32, shape=(1, 224, 384, 8))
Y = tf.placeholder(dtype=tf.float32, shape=(1, 224, 384, 2))

predict_flow2 = network.generator(X)
predict_flow2 = predict_flow2[1]


Y = further_resize_lbls(Y)

predict_flow2 = predict_flow2[:,:,:,0:2] 
loss_result = lhpl.endpoint_loss(Y,predict_flow2,1)
# loss_result = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y, predict_flow2))))

load_model_ckpt(sess,FLAGS.CKPT_FOLDER)

perform_testing()