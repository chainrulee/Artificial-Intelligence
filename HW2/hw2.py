####################################################################################
###  Import Libraries
####################################################################################
import cv2
import sys
import caffe
import numpy as np

####################################################################################
###  LOAD CAFFE MODEL
####################################################################################
model = "hw2-deploy.prototxt"
weights = "hw2-weights.caffemodel"

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
accuracy = 0
n = 0

# read input file
inputFile = "hw2-test-split.txt"

f = open(inputFile, 'r')
while True:
	line = f.readline()
	if not line:
		break
	a = line.split();
	print(a[0])

	# load image
	image = cv2.imread(a[0])

	# resize image to NN input dimensions
	inputResImg = cv2.resize(image, (nnInputWidth, nnInputHeight), interpolation=cv2.INTER_CUBIC)

	# this is to rearrange the data to match how caffe likes the input
	transposedInputImg = inputResImg.transpose((2,0,1))

	# put image into data blob
	net.blobs['data'].data[...]=transposedInputImg

	# do a forward pass
	out = net.forward()

	# get the predicted scores
	scores = out['prob']
	predict = np.argmax(scores)
	print("Predicted ID: {} = {}".format(predict, classes[predict]))
	print("True ID:      {} = {}".format(a[1], classes[int(a[1])]))
	if predict == int(a[1]):
		accuracy += 1
	n += 1
	print("Number of correct prediction = {}/{}".format(accuracy, n))

# calcuate accuracy
# number of test samples
n = 128
print("Accuracy = {}".format(float(accuracy)/128))




