import numpy as np
import os, sys
caffe_root = os.path.dirname(os.path.abspath(__file__))+'/../../'
sys.path.insert(0, caffe_root+'python')
#os.environ['GLOG_minloglevel'] = '2'
np.set_printoptions(linewidth=200)
import caffe
import cv2

if not os.path.isdir(caffe_root+'examples/images/CatWithMaskLMDB'):
	import subprocess
	with open(caffe_root+'examples/images/cat_with_mask.txt','w') as listfile:
		listfile.write('cat_with_mask.png 0')
	subprocess.check_output([caffe_root+'build/tools/convert_imageset',
			'--encoded=1',
			'--encode_type=png',
			'--gray=-1',
			'--check_colorchannels=4',
			caffe_root+'examples/images/',
			caffe_root+'examples/images/cat_with_mask.txt',
			caffe_root+'examples/images/CatWithMaskLMDB'])

caffe.set_mode_cpu()
nnet = caffe.Net(caffe_root+'examples/AFFINE_TEST/TestWithMask.prototxt', caffe.TRAIN)

def displayable(caffe3chanimage):
	return np.transpose(caffe3chanimage[:,:,:],(1,2,0)) / 255.0

def applymask(rgb, mask):
	reshaped = np.reshape(mask,(mask.shape[0],mask.shape[1],1))
	return np.multiply(rgb, np.repeat(reshaped,3,axis=2))

for ii in range(10000):

	nnet.forward()

	caffeim = nnet.blobs['data_rgb'].data
	
	rgbpart = displayable(caffeim[0,:3,:,:])
	mskpart = caffeim[0, 3,:,:].astype(np.uint8)
	flatmaskpart = mskpart.flatten()
	uniquevals = []
	for jj in range(flatmaskpart.size):
		if flatmaskpart[jj] not in uniquevals:
			uniquevals.append(flatmaskpart[jj])
	print("uniquevals == "+str(uniquevals))
	mskpart[mskpart == 0  ] = 70
	mskpart[mskpart == 255] = 0
	mskpart[mskpart == 100] = 255
	mskpart = mskpart.astype(np.float32) / 255.0
	
	cv2.imshow('rgb',rgbpart)
	cv2.imshow('mask',mskpart)
	cv2.imshow('masked',applymask(rgbpart,mskpart))
	cv2.waitKey(0)

