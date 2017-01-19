import numpy as np
import os, sys, time
caffe_root = os.path.dirname(os.path.abspath(__file__))+'/../../'
sys.path.insert(0, caffe_root+'python')
#os.environ['GLOG_minloglevel'] = '2'
np.set_printoptions(linewidth=200)
import caffe
import cv2

if not os.path.isdir(caffe_root+'examples/images/CatLMDB'):
	import subprocess
	with open(caffe_root+'examples/images/cat.txt','w') as listfile:
		listfile.write('cat.jpg 0\r\nfish-bike.jpg 0')
	subprocess.check_output([caffe_root+'build/tools/convert_imageset',
			'--encoded=1',
			'--encode_type=png',
			caffe_root+'examples/images/',
			caffe_root+'examples/images/cat.txt',
			caffe_root+'examples/images/CatLMDB'])

caffe.set_mode_cpu()
nnet = caffe.Net(caffe_root+'examples/AFFINE_TEST/Test.prototxt', caffe.TEST)

def displayable(caffeimage):
	return np.transpose(caffeimage,(1,2,0)) / 255.0

for ii in range(10000):

	beftime = time.time()
	nnet.forward()
	afttime = time.time()

	caffeim = nnet.blobs['data_rgb'].data

	print("nnet.blobs[conv1].data.shape == "+str(nnet.blobs['conv1'].data.shape)+" ... forward time: "+str(afttime - beftime)+" seconds")

	for jj in range(caffeim.shape[0]):
		cv2.imshow('caffe-im-'+str(jj), displayable(caffeim[jj,...]))
	cv2.waitKey(0)
