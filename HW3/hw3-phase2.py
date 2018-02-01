#!/usr/bin/env python

import os
from os import walk
import random

datasetDir = '/home/min/a/lee2832/ee570/hw3/images'
className = {0: 'fox', 1: 'bike', 2: 'elephant', 3: 'car'}

reducedRandomList = []
posNumKeep = 600
negNumKeep = 150

for key in className:
	imageList = []
	imageDir = datasetDir + '/' + className[key] + '/pos'
	for (dirpath, dirnames, filenames) in walk(imageDir):
		for f in filenames:
			imageList.append(imageDir + '/' + f + ' ' + str(key))
	reducedRandomList.extend(random.sample(imageList, posNumKeep))

for key in className:
	imageList = []
	imageDir = datasetDir + '/' + className[key] + '/neg'
	for (dirpath, dirnames, filenames) in walk(imageDir):	
		for f in filenames:
			imageList.append(imageDir + '/' + f + ' 4')
	reducedRandomList.extend(random.sample(imageList, negNumKeep))

# create file
f = file("hw3-data.txt", 'w')
for imageFile in reducedRandomList:
	f.write(imageFile + '\n')
f.close()

	



