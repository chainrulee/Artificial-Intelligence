require 'torch'
require 'nn'

numInputs = 4
numHiddenNodes = 4
includeBias = true
learningRate = 0.1
numEpochs = 25
thresholdLayer = false

-- read in data from txt file
function readData(dataFile, numRows, numColumns)
	local data = torch.Tensor(numRows, numColumns)
	local i = 1
	for line in io.lines(dataFile) do
		local input1, input2, input3, input4, label = line:match("([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)")
		data[i][1] = input1
		data[i][2] = input2
		data[i][3] = input3
		data[i][4] = input4
		data[i][5] = label
		i = i + 1
	end
	return data
end

trainData = readData("hw1-train-split.txt", 135, 5)
testData = readData("hw1-test-split.txt", 15, 5)

-- seperate the tensors into inputs and labels
trainInputs = trainData:narrow(2, 1, 4)
trainLabels = trainData:narrow(2, 5, 1)
testInputs = testData:narrow(2, 1, 4)
testLabels = testData:narrow(2, 5, 1) 

-- create a dataset from two tensors
function createDataset(inputs, labels)
	local numSamples = inputs:size(1)
	local dataset = { }
	for i=1, numSamples do
		local x = inputs:sub(i,i)
		local y = labels[i]
		dataset[i] = {x, y}
	end
	
	function dataset:size()
		return numSamples
	end
	return dataset
end

trainDataset = createDataset(trainInputs, trainLabels)
testDataset = createDataset(testInputs, testLabels)

--[[for index, data in ipairs(trainDataset) do
	print(index)
	for key, value in pairs(data) do
		print('\t', key, value)
	end
end]]

-- create the NN
myNN = nn.Sequential()

-- HIDDEN LAYER
myNN:add(nn.Linear(numInputs, numHiddenNodes, includeBias))
myNN:add(nn.Sigmoid())

-- OUTPUT LAYER
myNN:add(nn.Linear(numHiddenNodes, 3, includeBias))
myNN:add(nn.LogSoftMax())
print(myNN)

-- Train the network
criterion = nn.ClassNLLCriterion()
criterion.sizeAverage = false
trainer = nn.StochasticGradient(myNN,criterion)
trainer.learningRate = learningRate
trainer.maxIteration = numEpochs

-- Train
print("\nTraining the network.")
trainer:train(trainDataset)

--[[print("\nThis is what the weights/biases have trained to:")
print("Hidden Nodes Weights:")
print(myNN.modules[1].weight)
print("Hiddent Nodes Biases:")
print(myNN.modules[1].bias)
print("Output Nodes Weights:")
print(myNN.modules[3].weight)
print("Output Nodes Biases:")
print(myNN.modules[3].bias)]]

-- Test the network
totalLoss = 0
testSize = 15

print("\nTesting the network.")

for i=1,testSize do
	x = testDataset[i][1]
	d = testDataset[i][2]
	
	y = myNN:forward(x)
	if (thresholdLayer) then
		y = torch.round(y)
	end
	
--	print(d)
--	print(y)

	d = d[1]
	max, y = torch.max(y,2)
--	print(y[1][1])
--	print(d)

	if (y[1][1]~=d) then
		loss = 1
	else
		loss = 0
	end
	totalLoss = totalLoss + loss
	print("Predicted: "..y[1][1].."  Desired: "..d.." Correct: "..i-totalLoss.."/"..testSize)
--	print("totalLoss = ",totalLoss)
end

-- calculate average loss
averageLoss = totalLoss/testSize
print("Loss during test: " .. averageLoss .. "\n")

-- Save the network
-- create the filename
fileName = "hw1.torchModel"

-- Save the model
torch.save(fileName, myNN)

