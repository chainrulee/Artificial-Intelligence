#!/usr/bin/env python

import cv2
import os
from os import walk
import xml.etree.ElementTree as ET
import sys
sys.path.append('/home/min/a/ee570/edge-boxes-with-python/')
import edge_boxes

datasetDir = '/home/min/a/ee570/hw3-files/hw3-dataset'
className = {0: 'fox', 1: 'bike', 2: 'elephant', 3: 'car'}
classDirName = {0: 'n02119789', 1: 'n03792782', 2: 'n02504458', 3: 'n04037443'}

# Function to compute IoU
def bb_intersection_over_union(boxA, boxB):
        # determine if the boxes don't overlap at all
        if (boxB[0] > boxA[2]) or (boxA[0] > boxB[2]) or (boxB[1] > boxA[3]) or (boxA[1] > boxB[3]):
                return 0

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = (xB - xA + 1) * (yB - yA + 1)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

# check if a directory exists
if not os.path.exists("images"):
        os.makedirs("images")
if not os.path.exists("images/fox"):
	for key in className:
		os.makedirs("images/" + className[key])
		os.makedirs("images/" + className[key] + "/pos")
		os.makedirs("images/" + className[key] + "/neg")

for key in classDirName:
	# Get a list of all files in a directory
	imageDir = datasetDir + '/' + classDirName[key]
	imageFiles = []
	imageFilePath = []

	for (dirpath, dirnames, filenames) in walk(imageDir):
		imageFiles.extend(filenames)
		for fileName in filenames:
			imageFilePath.append(imageDir + '/' + fileName)

	print "Printing list of image files"
	print imageFiles
	print "num Image Files in " + classDirName[key], len(imageFiles)

	#imageFileNames = []
	#imageFileNames.append(imageDir + "/n02119789_10130.JPEG")
	windows = edge_boxes.get_windows(imageFilePath)

	imgNum = 1
	allPosNum = 0
	allNegNum = 0
	for imageFileName in imageFiles:
		print 'imgNum = ', imgNum
		# load the image
		image = cv2.imread(imageDir + '/' + imageFileName)
		#cv2.imshow('image', image)

		# load the annotations
		tree = ET.parse(datasetDir + "/annotation/" + classDirName[key] + '/' + imageFileName.replace('JPEG', 'xml'))
		objs = tree.findall('object')

		# This is one way of looping over the instances and extracting their coordinates
		posNum = 0;
		negNum = 0;
		#proNum = 1;
		#print 'Total proposal number: ' + str(len(windows[0]))
		for proBbox in windows[imgNum-1]:
			sampleBox = [proBbox[1], proBbox[0], proBbox[3], proBbox[2]]
			maxIou = 0;
			for ix, obj in enumerate(objs):
				bbox = obj.find('bndbox') # bndbox is a tag inside of "object".  It has the ground truth coordinates
				x1 = int(bbox.find('xmin').text)
				y1 = int(bbox.find('ymin').text)
				x2 = int(bbox.find('xmax').text)
				y2 = int(bbox.find('ymax').text)
				groundTruthBox = [x1, y1, x2, y2]
				iou = bb_intersection_over_union(sampleBox, groundTruthBox)		
				if iou > maxIou:
					maxIou = iou 
			if maxIou > 0.7 and posNum < 5:
				#print '###################################'
			#	print 'proposal ' + str(proNum) + '.'
			#	print 'IoU Score: ' + str(maxIou)  
				sample = image[int(proBbox[0]):int(proBbox[2])+1, int(proBbox[1]):int(proBbox[3])+1, :]
				#cv2.imshow('sample', sample)
				#cv2.waitKey(0)
				allPosNum += 1
				cv2.imwrite('images/' + className[key] + '/pos/image_' + str(allPosNum).zfill(4) + '.JPEG', sample)
				posNum += 1
				#break		
			elif maxIou < 0.4 and negNum < 5: 
				#print '###################################'
			#	print 'proposal ' + str(proNum) + '.'
			#	print 'IoU Score: ' + str(maxIou)
				sample = image[int(proBbox[0]):int(proBbox[2])+1, int(proBbox[1]):int(proBbox[3])+1, :]
				allNegNum += 1
				cv2.imwrite('images/' + className[key] + '/neg/image_' + str(allNegNum).zfill(4) + '.JPEG', sample)
				negNum += 1
			elif posNum + negNum is 10:
				break
			#else:
			#	print 'proposal ' + str(proNum) + '.'
			#	print 'IoU Score: ' + str(maxIou)	
			#proNum += 1
		imgNum += 1
		print 'posNum = ', posNum
		print 'negNum = ', negNum


