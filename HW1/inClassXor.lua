-------------------------------------------------------------------------------
--	Filename: inClassXor.lua
--	Author: Jared Johansen
-------------------------------------------------------------------------------
require 'torch'
require 'nn'

-------------------------------------------------------------------------------
--	Intro to "local"
-------------------------------------------------------------------------------
-- If you don't use local, the variable becomes global.  There is no scope limitation.
--[[
function test()
	x = 5
end
function test2()
	local y = 6
end

test()
test2()
--]]

-------------------------------------------------------------------------------
--	Intro to data types
-------------------------------------------------------------------------------
-- nil	Used to differentiate the value from having some data or no(nil) data.
-- boolean	Includes true and false as values. Generally used for condition checking.
-- number	Represents real(double precision floating point) numbers.
-- string	Represents array of characters.
-- function	Represents a method that is written in C or Lua.
-- table	Represent ordinary arrays, symbol tables, sets, records, graphs, trees, etc., and implements associative arrays. It can hold any value (except nil).
-- there are two other data types (userdata, thread)...but you won't be needing them.
-- You don't need to specify the type for most variables (function excluded).
-- We'll see examples of each data type

-- specify hyperparameters
numInputs      = 2
numHiddenNodes = 2
includeBias    = true
learningRate   = 0.1
numEpochs      = 25
thresholdLayer = false

-- specify data parameters
trainFile = '/home/min/a/jjohanse/ee570Lecture/train_split.txt'
testFile  = '/home/min/a/jjohanse/ee570Lecture/test_split.txt'
numSamples = 1000
trainRatio = 0.9

-- if true, additional output is printed to the console
debug = false

-- front matter
print("\nThis will demo a simple NN in torch to solve the XOR problem.")
print("There is a single hidden layer (fully-connected with sigmoid function).")
print("And an output layer (fully-connected with sigmoid function).")
print("Binary cross-entropy for Sigmoid is used as the loss function.")
print("Stochastic Gradient Descent is used in the backpropagation algorithm.")
print("Each epoch has 900 inputs.")
print("\nYou have chosen the following options:")
print("\tXOR Size:               " .. numInputs)
print("\tNodes:                  " .. numHiddenNodes)
print("\tInclude Bias:        ", includeBias)
print("\tLearning Rate:          " .. learningRate)
print("\tNumber of Epochs:       " .. numEpochs)
print("\tThreshold Layer:     ", thresholdLayer)


-------------------------------------------------------------------------------
--	Intro to torch operations (rand, round, sum, fmod, cat, sub)
--  Create the training/test data and labels
-------------------------------------------------------------------------------
-- create the inputs (1's or 0's)
inputs = torch.rand(numSamples, numInputs) -- generate a random matrix of numSamples (rows) x numInputs (columns)
inputs = torch.round(inputs)               -- round the inputs to 0 or 1

-- create the labels
summation = torch.sum(inputs, 2)     -- specify the torch tensor...and the dimension you want to sum over.  (1=columns, 2=rows)
labels = torch.fmod(summation,2)     -- perform the modulo operator (remainder after division). fmod(data, divisor).
together = torch.cat(inputs, labels) -- concatenate the two tensors

-- calculate train/test sizes
trainSize = numSamples * trainRatio
testSize = numSamples * (1-trainRatio)

-- split into train/test data
trainData = together:sub(1,trainSize) 			-- get a subtensor.  Starts in dimension 1 at index 1, goes through index trainSize
testData = together:sub(trainSize+1,numSamples) -- get a subtensor.  Starts in dimension 1 at trainSize+1, goes through index numSamples


-------------------------------------------------------------------------------
--	Intro to function
--	Save data to CSV file
-------------------------------------------------------------------------------
-- function that saves 2D data into a file, comma-separated
function saveDataToCsv(data, fileName)
	-- open file
	local out = io.open(fileName, "w")
	print("fileName:" .. fileName)

	-- loop over data
	for i=1,data:size(1) do
		for j=1,data:size(2) do
		
		    -- write data
		    out:write(data[i][j])
		    
		    -- if you've reached the end of your row, add a newline character
		    if j == data:size(2) then
		        out:write("\n") 
		    else -- add a comma
		        out:write(",")
		    end
		end
	end

	-- close your file
	out:close()
end

-- save data into file
saveDataToCsv(trainData, trainFile)
saveDataToCsv(testData, testFile)


-------------------------------------------------------------------------------
--	Intro to function
--	Read in data from a CSV file
-------------------------------------------------------------------------------
-- function that loads a csv file (with 2D data) into a tensor
function readDataFromCsv(fileName, numRows, numColumns)
	-- open file
	local csvFile = io.open(fileName, "r") 

	-- create tensor data structure
	local data = torch.Tensor(numRows, numColumns)

	-- keep track of line number
	local i = 1

	-- loop over file
	for line in io.lines(fileName) do  
		-- separate data from line
		local input1, input2, label = line:match("([^,]+),([^,]+),([^,]+)")

		-- store into tensor
		data[i][1] = input1
		data[i][2] = input2
		data[i][3] = label
		
		--increment
		i = i + 1
	end

	csvFile:close() 
	return data
end

trainData = readDataFromCsv(trainFile, 900, 3)
testData  = readDataFromCsv(testFile, 100, 3)


-------------------------------------------------------------------------------
--	Intro to tables
-------------------------------------------------------------------------------
-- The table type implements associative arrays. 
-- An associative array is an array that can be indexed not only with numbers, but also with strings or any other value of the language, except nil. 
-- Tables have no fixed size; you can add as many elements as you want to a table dynamically. 
--[[
myTable = {}     -- create a table and store its reference in `a'
k = "x"
myTable[k] = 10        -- new entry, with key="x" and value=10
myTable[20] = "great"  -- new entry, with key=20 and value="great"
print(myTable["x"])    --> 10
k = 20
print(myTable[k])      --> "great"
myTable["x"] = myTable["x"] + 1     -- increments entry "x"
print(myTable["x"])    --> 11
--]]

-------------------------------------------------------------------------------
--	Create a training and test dataset
-------------------------------------------------------------------------------
-- separate the tensors into inputs and labels
trainInputs = trainData:narrow(2, 1, 2) -- narrow dimension 2 from index 1 to index 2
trainLabels = trainData:narrow(2, 3, 1) -- narrow dimension 2 from index 3 to index 3

testInputs = testData:narrow(2, 1, 2)   -- narrow dimension 2 from index 1 to index 2
testLabels = testData:narrow(2, 3, 1)   -- narrow dimension 2 from index 3 to index 3

-- function that creates a dataset from two tensors
function createDataset(inputs, labels)
	-- figure out the size of the tensor
	local numSamples = inputs:size(1)
	
	-- create table
	local dataset = { }

	-- loop over tensors
	for i=1, numSamples do
		local x = inputs:sub(i,i)
		local y = labels:sub(i,i)
		dataset[i] = {x, y}
	end

	-- the train() function (used later) needs to have a dataset with a size() function.
	function dataset:size() 
		return numSamples 
	end

	return dataset
end

trainDataset = createDataset(trainInputs, trainLabels)
testDataset  = createDataset(testInputs, testLabels)


-------------------------------------------------------------------------------
--	Create the NN
-------------------------------------------------------------------------------
-- Neural networks are made up of several different pieces: 
--	the "shape" of the network, layers, activation functions, loss functions, training method
-- https://github.com/torch/nn/blob/master/doc/index.md

-- There are different types of NN shapes/structures.  Torch calls them "containers".
-- https://github.com/torch/nn/blob/master/doc/containers.md
-- We will use a simple sequential NN (i.e. full-connected, feed-forward)
myNN = nn.Sequential() 

-- The NN shape/structures are built using layers.  Torch calls them "layers".
-- https://github.com/torch/nn/blob/master/doc/simple.md
-- Most layers have an activation functions.  Torch calls them "transfer functions".
-- https://github.com/torch/nn/blob/master/doc/transfer.md

-- HIDDEN LAYER
myNN:add(nn.Linear(numInputs, numHiddenNodes, includeBias)) -- fully-connected layer
myNN:add(nn.Sigmoid())										-- sigmoid activiation function

-- OUTPUT LAYER
myNN:add(nn.Linear(numHiddenNodes, 1, includeBias)) -- fully-connected layer
myNN:add(nn.Sigmoid())								-- sigmoid activiation function

-- print out what the NN looks like
print("\nThis is what the NN looks like:")
print(myNN)
print("\nThis is what the weights/biases are initialized to:")
print("Hidden Nodes Weights:")
print(myNN.modules[1].weight)
print("Hidden Nodes Biases:")
print(myNN.modules[1].bias)
print("Output Nodes Weights:")
print(myNN.modules[3].weight)
print("Output Nodes Biases:")
print(myNN.modules[3].bias)

-------------------------------------------------------------------------------
--	Train the network
-------------------------------------------------------------------------------
-- Neural networks need a way to gauge if their prediction is "good" or not.
-- This gauge is called the "loss function".  Torch calls them "criterions".
-- There are lots of different types of loss functions.  Some people invent their own.
-- https://github.com/torch/nn/blob/master/doc/criterion.md#nn.BCECriterion
-- We will be using a binary cross-entropy for Sigmoid (two-class version of ClassNLLCriterion)
-- You'll want to use nn.ClassNLLCriterion() for your homework.
criterion = nn.BCECriterion()

-- Neural networks need a method by which to tweak the weights in the network (per the loss function).
-- This is called backpropagation.  Backpropagation is a method to take the loss, find out how
-- each weight contributed to the final result (whether they were contributing to the right output 
-- or the wrong) and "take corrective action".  They reward/punish each weight accordingly.  It is 
-- attempting to adjust the weights so that the network can make a better decision the next time it
-- sees a similar input.
-- While there are other alternatives out there, almost everyone uses stochastic gradient descent.
trainer = nn.StochasticGradient(myNN, criterion)

-- There are parameters (often called "hyperparameters") associated with training a network
-- https://github.com/torch/nn/blob/master/doc/training.md#nn.StochasticGradient
trainer.learningRate = learningRate
trainer.maxIteration = numEpochs

-- train
print("\nTraining the network.")
trainer:train(trainDataset)

print("\nThis is what the weights/biases have trained to:")
print("Hidden Nodes Weights:")
print(myNN.modules[1].weight)
print("Hidden Nodes Biases:")
print(myNN.modules[1].bias)
print("Output Nodes Weights:")
print(myNN.modules[3].weight)
print("Output Nodes Biases:")
print(myNN.modules[3].bias)

-------------------------------------------------------------------------------
--	Test the network
-------------------------------------------------------------------------------
-- keep track of the total loss (overall all the test data)
totalLoss = 0;

-- test
print("\nTesting the network.")

for i=1,testSize do
	-- get the input vector (x) and label (d)
	x = testDataset[i][1]
	d = testDataset[i][2]

	-- send it through the NN and get the output (y)
	y = myNN:forward(x)


	-- threshold the output
	if (thresholdLayer) then
		y = torch.round(y)
	end

	-- convert Tensors to numbers
	d = d[1][1]
	y = y[1][1]
	

	-- calculate the loss
	loss = y - d
	totalLoss = totalLoss + math.abs(loss)
	
	-- print the results of each layer
	if debug then
		print("Hidden Layer, Linear Output:")
		print(myNN.modules[1].output)

		print("Hidden Layer, Sigmoid Output:")
		print(myNN.modules[2].output)

		print("Output Layer, Linear Output:")
		print(myNN.modules[3].output)

		print("Output Layer, Sigmoid Output:")
		print(myNN.modules[4].output)
	end

	-- print predictions
	--print("Desired: " .. d .. ". Predicted: " .. y .. ". Loss: " .. loss)
end

-- calculate average loss
averageLoss = totalLoss/testSize
print("Loss during test: " .. averageLoss .. "\n")

-------------------------------------------------------------------------------
--	Save the network
-------------------------------------------------------------------------------
-- create the filename
fileName = "xor_" .. numInputs .. "input.torchModel"

--save the model
torch.save(fileName, myNN)

-- this is how you would load the model
-- model = torch.load(filename)


