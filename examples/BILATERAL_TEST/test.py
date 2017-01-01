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
nnet = caffe.Net(caffe_root+'examples/BILATERAL_TEST/Test.prototxt', caffe.TEST)

def displayable(caffeimage, idx=0):
	return np.round(np.transpose(caffeimage[idx,:,:,:],(1,2,0))).astype(np.uint8)

for ii in range(10000):

	beftime = time.time()
	nnet.forward()
	afttime = time.time()

	caffeim = nnet.blobs['data_rgb'].data
	filt_bilat = nnet.blobs['filt_bilat'].data

	print("forward time: "+str(afttime - beftime)+" seconds")

	for mbidx in range(caffeim.shape[0]):
		cv2.imshow(str(mbidx)+'caffeim', displayable(caffeim,mbidx))
		cv2.imshow(str(mbidx)+'filt_bilat', displayable(filt_bilat,mbidx))
	cv2.waitKey(0)

