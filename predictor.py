
import flow_test as ft
import helpers as hpl
import numpy as np
from   PIL import Image
import matplotlib as plt

# import synthetic_tf_converter as converter
import tensorflow as tf
import data_reader as dr
# import matplotlib.mlab as mlab
# import ijremote as ij
# import losses_helper as lhpl
folder = '../dataset_synthetic/driving/'
# folder = '/misc/lmbraid19/muazzama/dataset_synthetic/driving/'

img1 = folder + 'frames_finalpass_webp/35mm_focallength/scene_backwards/fast/left/0076.webp'
img2 = folder + 'frames_finalpass_webp/35mm_focallength/scene_backwards/fast/left/0077.webp'
disparity1 = folder + 'disparity/35mm_focallength/scene_backwards/fast/left/0076.pfm'
disparity2 = folder + 'disparity/35mm_focallength/scene_backwards/fast/left/0077.pfm'
opt_flow = folder + 'optical_flow/35mm_focallength/scene_backwards/fast/into_future/left/OpticalFlowIntoFuture_0076_L.pfm'
disp_change = folder + 'disparity_change/35mm_focallength/scene_backwards/fast/into_future/left/0076.pfm'

img3 = folder + 'frames_finalpass_webp/35mm_focallength/scene_backwards/fast/left/0107.webp'
opt_flow3 = folder + 'optical_flow/35mm_focallength/scene_backwards/fast/into_future/left/OpticalFlowIntoFuture_0106_L.pfm'
disp_change3 = folder + 'disparity_change/35mm_focallength/scene_backwards/fast/into_future/left/0106.pfm'
disparity3 = folder + 'disparity/35mm_focallength/scene_backwards/fast/left/0107.pfm'

''' ********************************************* this is the reading part ********************************************* '''
''' ********************************************* this is the reading part ********************************************* '''
''' ********************************************* this is the reading part ********************************************* '''
''' ********************************************* this is the reading part ********************************************* '''
''' ********************************************* this is the reading part ********************************************* '''
''' ********************************************* this is the reading part ********************************************* '''
# def show_optical_flow(label_batch): 

# 	factor = 0.4
# 	input_size = int(960 * factor), int(540 * factor)

# 	opt_u = np.squeeze(label_batch[:,:,:,0]) * input_size[0]
# 	opt_v = np.squeeze(label_batch[:,:,:,1]) * input_size[1]

# 	opt_u = opt_u.astype(np.uint8)
# 	opt_v = opt_v.astype(np.uint8)

# 	opt_u = Image.fromarray(opt_u) 
# 	opt_v = Image.fromarray(opt_v)


# 	opt_u.show()
# 	opt_v.show()

# sess = tf.InteractiveSession()
# img = tf.constant([[1,-2,3],[4,5,6],[-7,8,9]],dtype=tf.float32)
# warped = tf.constant([[1,2,-3],[4,0,-6],[-7,0,0]],dtype=tf.float32)

# print(sess.run(tf.abs(lh.get_occulation_aware_image(img,warped))))
# factor = 0.4
# input_size = int(960 * factor), int(540 * factor)

# features_train = dr.tf_record_input_pipelinev2(['one_record.tfrecords'])
# train_imageBatch, train_labelBatch = tf.train.shuffle_batch(
#                                         [features_train['input_n'], 
#                                         features_train['label_n']],
#                                         batch_size=1,
#                                         capacity=100,
#                                         num_threads=10,
#                                         min_after_dequeue=6)

# sess = tf.InteractiveSession()
# # summary_writer_train = tf.summary.FileWriter('./tbtest/')
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess, coord=coord)

# train_batch_xs, train_batch_ys = sess.run([train_imageBatch,train_labelBatch])
# # r1, r2 = np.split(train_batch_xs,2,axis=3)

# train_batch_ys = np.squeeze(train_batch_ys)
# # ij.setImage('loadedImage_u',train_batch_ys[:,:,0] * input_size[0])
# ij.setImage('loadedImage_v',train_batch_ys[:,:,1] * input_size[1])

# # imgg1 = np.squeeze(r1[:,:,:,0:3])
# # imgg2 = np.squeeze(r2[:,:,:,0:3]).astype(np.uint8)

# # flow = np.squeeze(train_batch_ys)
# # # Image.fromarray(flow[:,:,0]).save('n_opt_flow_u_load.tiff')
# # flow = predictor.denormalize_flow(flow,False)
# # Image.fromarray(flow).save('opf_2v.tiff')

# # Image.fromarray(flow[:,:,0]).save('test_flowu.tiff')
# # Image.fromarray(flow[:,:,1]).save('test_flowv.tiff')
# # flow = predictor.warp(imgg1,flow)

# # predictor.show_image(flow.eval()[0].astype(np.uint8),'warped_img')
# coord.request_stop()
# coord.join(threads)



''' ********************************************* from file example ********************************************* '''
''' ********************************************* from file example ********************************************* '''
''' ********************************************* from file example ********************************************* '''
''' ********************************************* from file example ********************************************* '''
''' ********************************************* from file example ********************************************* '''
''' ********************************************* from file example ********************************************* '''

predictor = ft.FlowPredictor()
predictor.preprocess(img1,img2,disparity1,disparity2)

# denormu = Image.open('flow_u1.tiff')
# denormv = Image.open('flow_v1.tiff')

# # data = np.dstack((denormu,denormv),dtype=np.float32)

# flow_gt = hpl.readPFM(opt_flow)[0]
factor = 0.4
input_size = int(960 * factor), int(540 * factor)

# converter = converter.SyntheticTFRecordsWriter()
# flow_gt = converter.downsample_opt_flow(flow_gt,input_size)
# ij.setImage('InputU',np.array(denormu))
# ij.setImage('InputV',np.array(denormv))

# flow_gt = np.delete(flow_gt,2,axis=2)
# flow_gt_u = flow_gt[:,:,0]
# flow_gt_v = flow_gt[:,:,1]


''' ********************************************* this is the writing part ********************************************* '''
''' ********************************************* this is the writing part ********************************************* '''
''' ********************************************* this is the writing part ********************************************* '''
''' ********************************************* this is the writing part ********************************************* '''
''' ********************************************* this is the writing part ********************************************* '''
''' ********************************************* this is the writing part ********************************************* '''

# def create_tf_example(self,patches,writer):

# 	for item in patches:

# 		# downsampled_opt_flow = self.downsample_labels(np.array(item['opt_fl']),2)
# 		# downsampled_disp_chng = self.downsample_labels(np.array(item['disp_chng']),0)

# 		width , height = item['depth'].shape[0] , item['depth'].shape[1]
# 		depth = item['depth'].tostring()
# 		depth2 = item['depth2'].tostring()

# 		opt_flow = item['optical_flow'].tostring()
# 		depth_chng = item['disp_change'].tostring()
# 		frames_finalpass_webp = item['web_p'].tostring()
# 		frames_finalpass_webp2 = item['web_p2'].tostring()




# 		example = tf.train.Example(features=tf.train.Features(
# 			feature={
# 				'width': self._int64_feature(width),
# 				'height': self._int64_feature(height),
# 				'depth1': self._bytes_feature(depth),
# 				'depth2': self._bytes_feature(depth2),
# 				'disp_chng': self._bytes_feature(depth_chng),
# 				'opt_flow': self._bytes_feature(opt_flow),
# 				'image1': self._bytes_feature(frames_finalpass_webp),
# 				'image2': self._bytes_feature(frames_finalpass_webp2)
# 		    }),
# 		)

# 		writer.write(example.SerializeToString())
# 		writer.close()


# converter = converter.SyntheticTFRecordsWriter()
# result = converter.from_paths_to_data(disparity1,
# 									  disparity2,
# 									  disp_change,
# 									  opt_flow,
# 									  img1,
# 									  img2,
# 									  1)



# train_writer = tf.python_io.TFRecordWriter('./one_record.tfrecords')
# create_tf_example(converter,result,train_writer)


# from here

# disp_change = hpl.readPFM(disp_change)[0]
# disp1 = hpl.readPFM(disparity1)[0]
# disp2 = hpl.readPFM(disparity2)[0]
# np.set_printoptions(threshold=np.nan)

# predictor = ft.FlowPredictor()
# predictor.preprocess(img1,img2,disparity1,disparity2)
# disp1 = hpl.readPFM(disparity1)[0]
# disp2 = hpl.readPFM(disparity2)[0]
# disp3 = hpl.readPFM(disparity3)[0]

# disp1 = Image.fromarray(disp1)
# disp2 = Image.fromarray(disp2)
# disp3 = Image.fromarray(disp3)


# # resize disparity values
# disp1 = disp1.resize(input_size,Image.NEAREST)
# disp2 = disp2.resize(input_size,Image.NEAREST)
# disp3 = disp2.resize(input_size,Image.NEAREST)


# lbl = predictor.read_gt(opt_flow,disp_change)
# lbl3 = predictor.read_gt(opt_flow3,disp_change)
# opt_flow = np.pad(lbl,((4,4),(0,0),(0,0)),'constant')
# opt_flow3 = np.pad(lbl3,((4,4),(0,0),(0,0)),'constant')

# disp2 = np.array(disp2)
# disp2 = np.pad(disp2,((4,4),(0,0)),'constant')

# disp3 = np.array(disp3)
# disp3 = np.pad(disp3,((4,4),(0,0)),'constant')

# opt_flow = tf.expand_dims(tf.convert_to_tensor(opt_flow,dtype=tf.float32),axis=0)
# opt_flow3 = tf.expand_dims(tf.convert_to_tensor(opt_flow3,dtype=tf.float32),axis=0)
# disp2 = tf.expand_dims(tf.convert_to_tensor(disp2,dtype=tf.float32),axis=0)
# disp2 = tf.expand_dims(tf.convert_to_tensor(disp2,dtype=tf.float32),axis=3)

# disp3 = tf.expand_dims(tf.convert_to_tensor(disp3,dtype=tf.float32),axis=0)
# disp3 = tf.expand_dims(tf.convert_to_tensor(disp3,dtype=tf.float32),axis=3)

# disp = tf.concat([disp2,disp3],axis=0)
# opt_flow = tf.concat([opt_flow,opt_flow3],axis=0)

# result = lhpl.flow_warp(disp,opt_flow)
# result = tf.squeeze(result)

# print(result)
# ij.setImage('depth',result[0].eval())
# ij.setImage('depth_change',disp_change)
# ij.setImage('depth1',disp1)
# ij.setImage('depth2',disp2)
predictor.predict()


# for testing with ground truth



# opt = hpl.readPFM(opt_flow3)[0]
# lbl = predictor.read_gt(opt_flow3,disp_change3)
# opt_flow3 = np.pad(lbl,((4,4),(0,0),(0,0)),'constant')
# predictor.postprocess(flow=opt_flow,show_flow=True,gt=True)

# print(opt_flow[:,:,0].shape)





# import losses_helper as lhpl

# img1 = Image.open(img1)
# img2 = Image.open(img2)
# img3 = Image.open(img3)
# img1 = img1.resize(input_size, Image.BILINEAR)
# img2 = img2.resize(input_size, Image.BILINEAR)
# img3 = img3.resize(input_size, Image.BILINEAR)

# img1 = np.array(img1,dtype=np.float32)
# img1 = np.pad(img1,((4,4),(0,0),(0,0)),'constant')

# img2 = np.array(img2,dtype=np.float32)
# img2 = np.pad(img2,((4,4),(0,0),(0,0)),'constant')

# img3 = np.array(img3,dtype=np.float32)
# img3 = np.pad(img3,((4,4),(0,0),(0,0)),'constant')

# opt_flow = tf.expand_dims(tf.convert_to_tensor(opt_flow,dtype=tf.float32),axis=0)
# opt_flow3 = tf.expand_dims(tf.convert_to_tensor(opt_flow3,dtype=tf.float32),axis=0)
# img1 = tf.expand_dims(tf.convert_to_tensor(np.array(img1),dtype=tf.float32),axis=0)
# img2 = tf.expand_dims(tf.convert_to_tensor(np.array(img2),dtype=tf.float32),axis=0)
# img3 = tf.expand_dims(tf.convert_to_tensor(np.array(img3),dtype=tf.float32),axis=0)


# img = tf.concat([img2,img3],axis=0)
# opt_flow = tf.concat([opt_flow,opt_flow3],axis=0)


# predictor.show_image(img2.eval()[0].astype(np.uint8),'normal_img')
# print(opt_flow)
# ij.setImage('flow',opt_flow.eval()[0,:,:,1])

# result = lhpl.flow_warp(img,opt_flow)



# resultt = lhpl.get_occulation_aware_image(img1,result)
# print(resultt)

# print(img1)
# # print(result.eval())
# predictor.show_image(img1.eval()[0].astype(np.uint8),'warped_img')
# predictor.show_image(resultt.eval()[0].astype(np.uint8),'warped_img')


# for testing

# Image.fromarray(opt.astype(np.uint8)).show()
# dispar1 = hpl.readPFM(disparity1)[0]
# dispar2 = hpl.readPFM(disparity2)[0]
# opt_flow = hpl.readPFM(opt_flow)[0]



# dispar_chng = hpl.readPFM(disp_change)[0]
# result1 = predictor.get_depth_from_disp(dispar1)
# result2 = predictor.get_depth_from_disp(dispar2)
# result3 = predictor.get_depth_from_disp(dispar_chng)
# result3 = predictor.get_depth_chng_from_disp_chng(dispar1,dispar_chng)
# Image.open(img1).show()
# Image.open(img2).show()
# Image.fromarray(result1).show()
# Image.fromarray(result2).show()
# Image.fromarray(result3).show()
# Image.fromarray(opt_flow[:,:,0]).show()
# Image.fromarray(opt_flow[:,:,1]).show()
# print(opt_flow[:,:,1])

# plt.hist(opt_flow[:,:,1], bins='auto')  # arguments are passed to np.histogram
# plt.title("Histogram with 'auto' bins")
# plt.show()