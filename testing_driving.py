import network
import tensorflow as tf

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


tf.app.flags.DEFINE_string('CKPT_FOLDER', 'ckpt/driving/epe/train/',
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

def parse_input(img1,img2,disp1,disp2):
	img1 = Image.open(img1)
	img2 = Image.open(img2)

	disp1 = Image.open(disp1)
	disp2 = Image.open(disp2)

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


img_pair, img2_orig = parse_input(FLAGS.IMG1,FLAGS.IMG2,FLAGS.DISPARITY1,FLAGS.DISPARITY2)


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