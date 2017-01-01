import os, sys, time
# force run on CPU?
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

caffe_root = os.path.dirname(os.path.abspath(__file__))+'/../../'
sys.path.insert(0, caffe_root+'python')

#os.environ['GLOG_minloglevel'] = '2'

import numpy as np
np.set_printoptions(linewidth=200)
import cv2
import caffe


if not os.path.isdir(caffe_root+'examples/images/CatLMDB'):
	import subprocess
	with open(caffe_root+'examples/images/cat.txt','w') as listfile:
		listfile.write('cat.jpg 0')
	subprocess.check_output([caffe_root+'build/tools/convert_imageset',
			'--encoded=1',
			'--encode_type=png',
			caffe_root+'examples/images/',
			caffe_root+'examples/images/cat.txt',
			caffe_root+'examples/images/CatLMDB'])

caffe.set_mode_gpu()
nnet = caffe.Net(caffe_root+'examples/BILATERAL_TEST/Test_meanfield.prototxt', caffe.TEST)

def displayable(caffeimage):
	return np.transpose(caffeimage[0,:,:,:],(1,2,0)) / 255.0

for ii in range(10000):

	beftime = time.time()
	nnet.forward()
	afttime = time.time()

	caffeim = nnet.blobs['data_rgb'].data
	#filt_space = nnet.blobs['filt_space'].data
	filt_bilat = nnet.blobs['pred'].data / 3.0
	# divide by 3 because: 1 from orig, + 2 iterations, all summed without softmax

	print("forward time: "+str(afttime - beftime)+" seconds")

	cv2.imshow('caffeim', displayable(caffeim))
	#cv2.imshow('filt_space', displayable(filt_space))
	cv2.imshow('filt_bilat', displayable(filt_bilat))
	cv2.waitKey(0)

