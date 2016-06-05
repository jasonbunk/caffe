import numpy as np
import os, sys
caffe_root = os.path.dirname(os.path.abspath(__file__))+'/../../'
sys.path.insert(0, caffe_root+'python')
#os.environ['GLOG_minloglevel'] = '2'
np.set_printoptions(linewidth=200)
import caffe
import cv2

if not os.path.isdir(caffe_root+'examples/images/CatLMDB'):
	import subprocess
	subprocess.check_output([caffe_root+'build/tools/convert_imageset',
			'--encoded=1',
			'--encode_type=png',
			caffe_root+'examples/images/',
			caffe_root+'examples/images/cat.txt',
			caffe_root+'examples/images/CatLMDB'])

caffe.set_mode_cpu()
nnet = caffe.Net(caffe_root+'examples/AFFINE_TEST/test.prototxt', caffe.TRAIN)

def displayable(caffeimage):
	return np.transpose(caffeimage[0,:,:,:],(1,2,0)) / 255.0

for ii in range(10000):

	nnet.forward()

	caffeim = nnet.blobs['data_rgb'].data

	cv2.imshow('caffe-im', displayable(caffeim))
	cv2.waitKey(0)
