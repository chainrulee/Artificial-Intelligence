#!/usr/bin/env python

#################################################################################################
# Import libraries
# These are libraries that you will want to use for this homework
# cv2 is a libary that is useful for image manipulation
# os is a libary that does system type things (like walking directories)
# xml... is a library that lets you parse XML files easily (the annotations are stored in xml files)
# the next three are necessary to import the edge_boxes library that we'll be using
# random helps you sample things in a random fashion
# caffe lets you use caffe
#################################################################################################
import cv2
import os
from os import walk
import sys
sys.path.append('/home/min/a/ee570/edge-boxes-with-python/')
import edge_boxes
import random
import caffe
import numpy as np
from numpy import inf

datasetDir = '/home/min/a/ee570/hw3-files/hw3-dataset'
className = {0: 'fox', 1: 'bike', 2: 'elephant', 3: 'car'}
classDirName = {0: 'n02119789', 1: 'n03792782', 2: 'n02504458', 3: 'n04037443'}
classNum = {0: 324, 1: 415, 2: 518, 3: 427}

####################################################################################
###  LOAD CAFFE MODEL
####################################################################################
model = "hw3-deploy.prototxt"
weights = "hw3-weights.caffemodel"

# specify caffe mode of operation
caffe.set_mode_cpu()

# create net object
print("Create Net Object")
net = caffe.Net(model, weights, caffe.TEST)

classes = ["Fox", "Bike", "Elephant", "Car"]
nnInputWidth = 32
nnInputHeight = 32

####################################################################################
### FEED IMAGES AND GET THE OUTPUTS
####################################################################################
# randomly choose one image from the dataset
key = random.randint(0,3)
imageDir = datasetDir + '/' +  classDirName[key]
imageFiles = []

for (dirpath, dirnames, filenames) in walk(imageDir):
	for fileName in filenames:
		imageFiles.append(imageDir + '/' + fileName)	
	
numImage = random.randint(0, classNum[key]-1)
imageFileName = imageFiles[numImage]
maxScore = -inf;

# load image
image = cv2.imread(imageFileName)

# get proposals
imageFileNames = []
imageFileNames.append(imageFileName)
windows = edge_boxes.get_windows(imageFileNames)

print "Number of proposals:", len(windows[0])
proposal = 1;
maxProposal = 0;

for proBbox in windows[0]:
	sampleBox = [proBbox[1], proBbox[0], proBbox[3], proBbox[2]]
	sample = image[int(proBbox[0]):int(proBbox[2])+1, int(proBbox[1]):int(proBbox[3])+1, :]

	# resize image to NN input dimensions
	inputResImg = cv2.resize(sample, (nnInputWidth, nnInputHeight), interpolation=cv2.INTER_CUBIC)

	# this is to rearrange the data to match how caffe likes the input
	transposedInputImg = inputResImg.transpose((2,0,1))

	# put image into data blob
	net.blobs['data'].data[...]=transposedInputImg

	# do a forward pass
	out = net.forward()

	# get the predicted scores
	scores = out['prob']
	if np.amax(scores) > maxScore and np.argmax(scores) != 4:
		maxScore = np.amax(scores)
		predict = np.argmax(scores)
		maxProposal = proposal
		x1 = int(proBbox[1])
		x2 = int(proBbox[3])
		y1 = int(proBbox[0])
		y2 = int(proBbox[2])
		
	proposal += 1;

print "Best Propsoal Index:", maxProposal
print("Predicted ID: {} = {}".format(predict, classes[predict]))
print("True ID:      {} = {}".format(key, className[key]))
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
cv2.imshow('origImage', image)
print "Press the space bar to exit"
cv2.waitKey(0)





