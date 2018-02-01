-------------------------------------------------------------------------------
--	Filename: calculateValues.lua
--	Author: Jared Johansen
-------------------------------------------------------------------------------


--		Input	Hidden Layer	Output Layer		Output
--
--		A ------- N1 ----------- 
--		  \	/               \
--		   \   /                 \
--		    \ /	                  \
--		     X                   N3 ------------- Y
--	            / \	                 /
--		   /   \                /
--		  /	\              /
--		B ------ N2 -----------
--

-------------------------------------------------------------------------------
--	Define the Inputs and Weights
-------------------------------------------------------------------------------
inputA = 1
inputB = 1
useInitializedValues = 0

if (useInitializedValues == 1) then
	--------- Initialized values ---------
	weight_A_to_N1 = -0.0554	
	weight_B_to_N1 = -0.6980
	N1_bias = -0.1413

	weight_A_to_N2 = -0.4591
	weight_B_to_N2 =  0.5224
	N2_bias = -0.6401

	weight_N1_to_N3 =  0.6868
	weight_N2_to_N3 = -0.0791
	N3_bias = 0.6758
else
	--------- Trained values ---------
	weight_A_to_N1 =  7.4725
	weight_B_to_N1 = -7.6722
	N1_bias        = -4.0516

	weight_A_to_N2 = -7.5305
	weight_B_to_N2 =  7.3055
	N2_bias        = -3.9622

	weight_N1_to_N3 =  14.5551
	weight_N2_to_N3 =  14.5385
	N3_bias         = -7.1691
end

--  myNN.modules[1].weight:
--	weight_A_to_N1  weight_B_to_N1
--  weight_A_to_N2  weight_B_to_N2

-------------------------------------------------------------------------------
--	Calculate values
-------------------------------------------------------------------------------
N1_input = (inputA * weight_A_to_N1) + (inputB * weight_B_to_N1) + N1_bias
N2_input = (inputA * weight_A_to_N2) + (inputB * weight_B_to_N2) + N2_bias

N1_output = 1 / (1 + 2.71828^(-N1_input)) -- this is the sigmoid function: 1/(1+e^-x)
N2_output = 1 / (1 + 2.71828^(-N2_input)) -- this is the sigmoid function: 1/(1+e^-x)

N3_input = (N1_output * weight_N1_to_N3) + (N2_output * weight_N2_to_N3) + N3_bias
N3_output = 1 / (1 + 2.71828^(-N3_input)) -- this is the sigmoid function: 1/(1+e^-x)

-------------------------------------------------------------------------------
--	Print values
-------------------------------------------------------------------------------
print("inputA:      ", inputA)
print("inputB:      ", inputB)
print("N1_input:    ", N1_input)
print("N2_input:    ", N2_input)
print("N1_output:   ", N1_output)
print("N2_output:   ", N2_output)
print("N3_input:    ", N3_input)
print("N3_output:   ", N3_output)

