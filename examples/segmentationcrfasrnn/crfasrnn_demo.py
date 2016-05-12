# -*- coding: utf-8 -*-
"""
This package contains code for the "CRF-RNN" semantic image segmentation method, published in the 
ICCV 2015 paper Conditional Random Fields as Recurrent Neural Networks. Our software is built on 
top of the Caffe deep learning library.
Contact:
Shuai Zheng (szheng@robots.ox.ac.uk), Sadeep Jayasumana (sadeep@robots.ox.ac.uk), Bernardino Romera-Paredes (bernard@robots.ox.ac.uk)
Supervisor: 
Philip Torr (philip.torr@eng.ox.ac.uk)
For more information about CRF-RNN, please vist the project website http://crfasrnn.torr.vision.
"""
caffe_root = '../../'
import sys
sys.path.insert(0, caffe_root+'python')
import os, time
import cPickle
import logging
import numpy as np
from PIL import Image as PILImage
import cStringIO as StringIO
import caffe

MODEL_FILE = 'TVG_CRFRNN_new_deploy.prototxt'
PRETRAINED = 'TVG_CRFRNN_COCO_VOC.caffemodel'
IMAGE_FILE = 'input.jpg'

net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
caffe.set_mode_gpu()
caffe.set_device(0)
input_image = 255 * caffe.io.load_image(IMAGE_FILE)
image = PILImage.fromarray(np.uint8(input_image))
image = np.array(image)

pallete = [0,0,0,  128,0,0,  0,128,0,  128,128,0,  0,0,128,  128,0,128,  0,128,128,  128,128,128,  64,0,0,  192,0,0,  64,128,0,  192,128,0,  64,0,128,  192,0,128,  64,128,128,  192,128,128,  0,64,0,  128,64,0,  0,192,0,  128,192,0,  0,64,128,  128,64,128,  0,192,128,  128,192,128,  64,64,0,  192,64,0,  64,192,0,  192,192,0]

mean_vec = np.array([103.939, 116.779, 123.68], dtype=np.float32)
reshaped_mean_vec = mean_vec.reshape(1, 1, 3) # in BGR order

# Rearrange channels to form BGR
im = image[:,:,::-1]
# Subtract mean
im = im - reshaped_mean_vec

# Pad as necessary
cur_h, cur_w, cur_c = im.shape
pad_h = 500 - cur_h
pad_w = 500 - cur_w
im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode = 'constant', constant_values = 0)

im = im.transpose((2,0,1))
net.blobs['data'].data[...] = im

# Get predictions
begintime = time.time()
net.forward()
endtime = time.time()
print("prediction time: "+str(endtime-begintime)+" sec")


out = net.blobs['pred'].data[0].argmax(axis=0).astype(np.uint8)
seg2 = out[0:cur_h,0:cur_w]
output_im = PILImage.fromarray(seg2)
output_im.putpalette(pallete)
output_im.save('output.png')

