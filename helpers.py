import re
import numpy as np
import tensorflow as tf 

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode('utf-8').rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


# warp the flow values to the image.
def warp(img,flow):
    x = list(range(0,self.input_size[0]))
    y = list(range(0,self.input_size[1] + 8))
    X, Y = tf.meshgrid(x, y)

    X = tf.cast(X,np.float32) + flow[:,:,0]
    Y = tf.cast(Y,np.float32) + flow[:,:,1]

    con = tf.stack([X,Y])
    result = tf.transpose(con,[1,2,0])
    result = tf.expand_dims(result,0)
    return tf.contrib.resampler.resampler(img[np.newaxis,:,:,:],result)



def swap_images_for_back_flow(images):

    '''
        Assuming batch_size = 32

        take half the batch_size and swap the images such that from image 1 to 16 
        batch contains I1,I2 and image 16 to 32 it contains I2,I1.
    '''

    first_rgbd = images[:,:,:,0:4]
    second_rgbd = images[:,:,:,4:8]

    return tf.concat([second_rgbd,first_rgbd],axis=3)