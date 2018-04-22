import run_network
import data_reader
import multiprocessing
# # read data
# img1 = Image.open(os.path.join(examples_dir,'sculpture1.png'))
# img2 = Image.open(os.path.join(examples_dir,'sculpture2.png'))

# input_data = prepare_input_data(img1,img2,data_format)

# print(get_available_gpus())
prefix = '../dataset_synthetic/'

# # with tf.name_scope("datareader"):
train_filenames = [
				prefix+'driving_TRAIN.tfrecords',
				prefix+'flying_TRAIN.tfrecords',
				prefix+'monkaa_TRAIN.tfrecords'
			]

test_filenames = [
				prefix+'driving_TEST.tfrecords'
				# prefix+'flyingthings3d_TEST.tfrecords',
				# prefix+'monkaa_TEST.tfrecords'
]





train_features = data_reader.tf_record_input_pipeline(train_filenames,version='1')
# test_features = data_reader.tf_record_input_pipeline(test_filenames,version='2')

reader = run_network.DatasetReader()
reader.train(train_features,'test_features')
# # reader.main(train_features,test_features)
# # print(helpers.readPFM('0006.pfm'))