//Multi-layer perceptron with back propagation
//Author: Kieron

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <cmath>

//Initialise constants
const int INPUT_NODES = 46;				//5 rows, 9 columns, 1 bias, total inputs of 46
const int NUMBER_OF_LETTERS = 46;		//5 for each vowel, then 1 for the other 21 letters
const int NUMBER_TEST_LETTERS = 130;	//5 test patterns for each letter
const int TARGET_SIZE = 5;				//our target is a vector of size 5, i.e 0,0,1,0,0 for 'i'
const int HIDDEN_NODES = 23;			//Rule of thumb is half the number of inputs
const int OUTPUT_NODES = 5;				//our output will be a vector of size 5, same reason as the TARGET_SIZE
const float TRAINING_SETS = (46 * 5);	//5 output nodes for each letter, there are 46 letters total in training (can change)
const float bias = 1.0f;				//bias
const float minError = 0.01f;			//minimum accepted error
const float gain = 0.1f;				//prevents stagnation

//Declare Function Prototypes
void InitialiseWeightMatrix(std::vector< std::vector<float> > &, int, int);
void ReadInputFromFile(std::ifstream &, std::vector< std::vector<float> > &, std::vector< std::vector<float> > &);
void ReadTargets(std::ifstream &, std::vector< std::vector<float> > &, int);
void ReadTestInputFromFile(std::ifstream &, std::vector< std::vector<float> > &);
void TrainingPhase(std::ifstream &, std::vector< std::vector<float> > &, std::vector< std::vector<float> > &,
					std::vector<float> &, std::vector<float> &,
					std::vector<float> &, std::vector<float> &,
					std::vector<float> &, std::vector<float> &,
					std::vector< std::vector<float> > &, 
					std::vector< std::vector<float> > &, std::vector<float> &);
void PropogateHiddenLayer(int, std::vector<float> &, std::vector<float> &, std::vector< std::vector<float> > &);
void ActivationFunction(std::vector<float> &, std::vector<float> &, int);
void PropogateOutputLayer(std::vector<float> &, std::vector<float> &, std::vector< std::vector<float> > &);
void CalculateDifferences(int, std::vector<float> &, std::vector<float> &, std::vector<float> &);
void SumDifferences(float &, std::vector<float> &);
void CalculateOuterLayerError(std::vector<float> &, std::vector<float> &, std::vector<float> &);
void CalculateHiddenLayerError(std::vector<float> &, std::vector<float> &, std::vector<float> &,
	std::vector<float> &, std::vector< std::vector<float> > &);
void UpdateHiddenWeightMatrix(std::vector< std::vector<float> > &, std::vector<float> &, std::vector<float> &);
void UpdateInputWeightMatrix(int, std::vector< std::vector<float> > &, std::vector<float> &, std::vector< std::vector<float> > &);
void CalculateAverageDifference(float &, float &);
void TestingPhase(std::ifstream &, std::ofstream &, std::vector< std::vector<float> > &,
	std::vector<float> &, std::vector<float> &, std::vector<float> &,
	std::vector<float> &, std::vector< std::vector<float> > &,
	std::vector< std::vector<float> > &, std::vector<int> &);

int main()
{
	std::vector< std::vector<float> > trainingInputNeurons(NUMBER_OF_LETTERS, std::vector<float>(INPUT_NODES));		//NOTE: 46 input nodes per pattern, 46 patterns for training, 130 for testing
	std::vector< std::vector<float> > testingInputNeurons(NUMBER_TEST_LETTERS, std::vector<float>(INPUT_NODES));		//may be different size to training
	std::vector<float> hiddenNeurons(HIDDEN_NODES);
	std::vector<float> outputNeurons(OUTPUT_NODES);
	std::vector< std::vector<float> > target(NUMBER_OF_LETTERS, std::vector<float>(TARGET_SIZE));					//the targets that each input pattern corresponds to
	std::vector<float> hiddenNet(HIDDEN_NODES);				//stores summations
	std::vector<float> outputNet(OUTPUT_NODES);
	std::vector<float> hiddenActOutput(HIDDEN_NODES);		//actual output for each hidden node
	std::vector<float> outputActOutput(OUTPUT_NODES);		//actual output for each output node
	std::vector<float> differences(OUTPUT_NODES);
	std::vector<float> hiddenLayerError(HIDDEN_NODES);
	std::vector<float> outerLayerError(OUTPUT_NODES);
	std::vector<int> results(OUTPUT_NODES);
	std::vector< std::vector<float> > inputHiddenWeights(HIDDEN_NODES, std::vector<float>(INPUT_NODES));		//weight matrix for input to hidden layer nodes
	std::vector< std::vector<float> > hiddenOutputWeights(OUTPUT_NODES, std::vector<float>(HIDDEN_NODES+1));		//weight matrix for hidden to output layer nodes

	std::ifstream trainingFile;
	std::ifstream testingFile;
	std::ofstream outputFile;
	bool trainingFlag = true;

	InitialiseWeightMatrix(inputHiddenWeights, HIDDEN_NODES, INPUT_NODES);
	InitialiseWeightMatrix(hiddenOutputWeights, OUTPUT_NODES, (HIDDEN_NODES+1));

	std::cout << "Beginning Training..." << std::endl;
	TrainingPhase(trainingFile, trainingInputNeurons, target, hiddenNet, outputNet, hiddenActOutput, outputActOutput,
		hiddenLayerError, outerLayerError, inputHiddenWeights, hiddenOutputWeights, differences);
	std::cout << "Finishing Training..." << std::endl;

	std::cout << "Beginning Testing..." << std::endl;
	TestingPhase(testingFile, outputFile, testingInputNeurons, hiddenNet, outputNet, hiddenActOutput, outputActOutput,
		inputHiddenWeights, hiddenOutputWeights, results);
	std::cout << "Finishing Testing..." << std::endl;


	getchar();
	return 0;
}

void InitialiseWeightMatrix(std::vector< std::vector<float> > &weight, int rows, int columns)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			//initialise weight matrix with random values between -0.5 and 0.5
			float temp = (((float)rand() / ((float)(RAND_MAX)+(float)(1))) - 0.5f);
			weight[i][j] = temp;
		}
	}
}

void ReadInputFromFile(std::ifstream &trainingFile, std::vector< std::vector<float> > &input, 
	std::vector< std::vector<float> > &target)
{
	std::string temp = "";
	for (int i = 0; i < NUMBER_OF_LETTERS; i++)		//reading in a letter
	{
		for (int j = 0; j < INPUT_NODES; j++)		//each letter has 46 input nodes
		{
			if (j == 0)		//inserting bias into first input node
			{
				input[i][j] = bias;
			}
			else		//inserting values after bias
			{
				std::getline(trainingFile, temp, ' ');
				std::istringstream ss(temp);
				input[i][j] = std::stoi(temp);
			}
		}
		ReadTargets(trainingFile, target, i);
	}
}

void ReadTargets(std::ifstream& trainingFile, std::vector< std::vector<float> > &target, int index)
{
	std::string temp = "";
	for (int i = 0; i < TARGET_SIZE; i++)		//reading targets into target vector
	{
		std::getline(trainingFile, temp, ' ');
		std::istringstream ss(temp);
		target[index][i] = std::stoi(temp);
	}
}

void ReadTestInputFromFile(std::ifstream &testingFile, std::vector< std::vector<float> > &input)
{
	std::string temp = "";
	for (int i = 0; i < NUMBER_TEST_LETTERS; i++)		//reading in a letter
	{
		for (int j = 0; j < INPUT_NODES; j++)		//each letter has 46 input nodes
		{
			if (j == 0)		//inserting bias into first input node
			{
				input[i][j] = bias;
			}
			else		//inserting values after bias
			{
				std::getline(testingFile, temp, ' ');
				std::istringstream ss(temp);
				input[i][j] = std::stoi(temp);
			}
		}
	}
}

void PropogateHiddenLayer(int charInd, std::vector<float> &hiddenNet, std::vector< std::vector<float> > &input,
	std::vector< std::vector<float> > &inputHiddenWeights)
{
	for (int i = 0; i < HIDDEN_NODES; i++)	
	{
		for (int j = 0; j < INPUT_NODES; j++)
		{
			hiddenNet[i] += (input[charInd][j] * inputHiddenWeights[i][j]);		//summation of bias and input
		}
	}
}

void ActivationFunction(std::vector<float> &actualOutput, std::vector<float> &netVector, int numOfNodes)
{
	for (int i = 0; i < numOfNodes; i++)
	{
		actualOutput[i] = 1.0 / (1.0 + (exp(-netVector[i])));
	}
}

void PropogateOutputLayer(std::vector<float> &outputNet, std::vector<float> &hiddenActOutput,
	std::vector< std::vector<float> > &hiddenOutputWeights)
{
	for (int i = 0; i < OUTPUT_NODES; i++)
	{
		for (int j = 0; j < (HIDDEN_NODES+1); j++)
		{
			if (j == 0)
			{
				outputNet[i] +=  (bias * hiddenOutputWeights[i][j]);		//include a bias
			}
			else
			{
				outputNet[i] += (hiddenActOutput[j-1] * hiddenOutputWeights[i][j]);		//summation of hidden layer's outputs
			}	
		}
	}
}

void CalculateDifferences(int charIndex, std::vector<float> &diff, std::vector< std::vector<float> > &targets, std::vector<float> &outputActOutput)
{
	for (int i = 0; i < OUTPUT_NODES; i++)
	{
		diff[i] = targets[charIndex][i] - outputActOutput[i];
	}
}

void SumDifferences(float &sumDiff, std::vector<float> &diff)
{
	for (int i = 0; i < OUTPUT_NODES; i++)
	{
		sumDiff += diff[i];
	}
}

void CalculateOuterLayerError(std::vector<float> &outerLayerError, std::vector<float> &outputActOutput, std::vector<float> &diff)
{
	for (int i = 0; i < OUTPUT_NODES; i++)
	{
		outerLayerError[i] = outputActOutput[i] * (1 - outputActOutput[i]) * diff[i];
	}
}

void CalculateHiddenLayerError(std::vector<float> &hiddenLayerError, std::vector<float> &outerLayerError, 
	std::vector<float> &hiddenActOutput, std::vector<float> &diff, 
	std::vector< std::vector<float> > &hiddenOutputWeights)
{
	std::vector<float> errorSum(HIDDEN_NODES);
	
	for (int i = 0; i < OUTPUT_NODES; i++)
	{
		for (int j = 1; j < (HIDDEN_NODES+1); j++)
		{
			errorSum[j-1] += outerLayerError[i] * hiddenOutputWeights[i][j];
		}
	}
	for (int i = 0; i < HIDDEN_NODES; i++)
	{
		hiddenLayerError[i] = hiddenActOutput[i] * (1 - hiddenActOutput[i]) * errorSum[i];
	}
}

void UpdateHiddenWeightMatrix(std::vector< std::vector<float> > &weightMatrix, std::vector<float> &error, 
	std::vector<float> &inputFromPreviousLayer)
{
	for (int i = 0; i < OUTPUT_NODES; i++)
	{
		for (int j = 0; j < (HIDDEN_NODES+1); j++)
		{
			if (j == 0) //inserting bias into weight matrix
			{
				weightMatrix[i][j] += (bias * gain * error[i]);
			}
			else
			{
				weightMatrix[i][j] += (inputFromPreviousLayer[j-1] * gain * error[i]);
			}
		}
	}
}

void UpdateInputWeightMatrix(int index, std::vector< std::vector<float> > &weightMatrix, std::vector<float> &error,
	std::vector< std::vector<float> > &input)
{
	for (int i = 0; i < HIDDEN_NODES; i++)
	{
		for (int j = 0; j < (INPUT_NODES); j++)
		{
			if (j == 0) //inserting bias into weight matrix
			{
				weightMatrix[i][j] += (bias * gain * error[i]);
			}
			else
			{
				weightMatrix[i][j] += (input[index][j] * gain * error[i]);
			}
		}
	}
}

void CalculateAverageDifference(float &aveDiff, float &sumDiff)
{
	aveDiff = sumDiff / TRAINING_SETS;
}

void TrainingPhase(std::ifstream &trainingFile, std::vector< std::vector<float> > &input, std::vector< std::vector<float> > &target,
	std::vector<float> &hiddenNet, std::vector<float> &outputNet, 
	std::vector<float> &hiddenActOutput, std::vector<float> &outputActOutput,
	std::vector<float> &hiddenLayerError, std::vector<float> &outerLayerError, 
	std::vector< std::vector<float> > &inputHiddenWeights,
	std::vector< std::vector<float> > &hiddenOutputWeights, std::vector<float> &diff)
{
	bool trainingFlag = true;
	float sumDiff = 0.0, aveDiff = 0.0;
	int epoch = 0;
	
	trainingFile.open("alphabetTraining.txt");
	ReadInputFromFile(trainingFile, input, target);		//reading in inputs and targets
	trainingFile.close();
	
	//training 
	while (trainingFlag)
	{		
		for (int i = 0; i < NUMBER_OF_LETTERS; i++)	//go through each character in input, training the MLP
		{
			//reset hidden node net values, output node net values, 
			//actual output of hidden and output layers, and error vectors to 0
			std::fill(hiddenNet.begin(), hiddenNet.end(), 0);
			std::fill(outputNet.begin(), outputNet.end(), 0);
			std::fill(hiddenActOutput.begin(), hiddenActOutput.end(), 0);
			std::fill(outputActOutput.begin(), outputActOutput.end(), 0);
			std::fill(hiddenLayerError.begin(), hiddenLayerError.end(), 0);
			std::fill(outerLayerError.begin(), outerLayerError.end(), 0);
			//calculate net values of the hidden layer nodes
			//net values are calculated by summing each input node
			//multipled by their corresponding weights in the weight matrix
			PropogateHiddenLayer(i, hiddenNet, input, inputHiddenWeights);		//summation of bias and inputs
																				//put hidden node net values through sigmoid activation function to get their output
																				//to get the output from the hidden nodes that are inputs to the outer layer node we must use the
																				//sigmoid function				
			ActivationFunction(hiddenActOutput, hiddenNet, HIDDEN_NODES);

			//calculate net value of outer layer node
			//this calculation uses the bias and the outputs from the two hidden layer nodes
			PropogateOutputLayer(outputNet, hiddenActOutput, hiddenOutputWeights);	//calculate output of outer layer node using sigmoid activation function
																					//using the net value just calculated we can use the sigmoid function on it to get our actual output 
																					//of the outer layer node
			ActivationFunction(outputActOutput, outputNet, OUTPUT_NODES);
				
			//calculate difference between target and actual output
			CalculateDifferences(i, diff, target, outputActOutput);

			//sum the target difference
			SumDifferences(sumDiff, diff);
				
			//calculate error for outer layer node (equation 9)
			CalculateOuterLayerError(outerLayerError, outputActOutput, diff);
				
			//calculate error for hidden layer nodes (equation 10), no summation required as only one output
			CalculateHiddenLayerError(hiddenLayerError, outerLayerError, hiddenActOutput, diff, hiddenOutputWeights);
				
			//Back propagation occurring to update weights based on error values
			//update weights between hidden and outer layer
			UpdateHiddenWeightMatrix(hiddenOutputWeights, outerLayerError, hiddenActOutput);
			//update weights between input and hidden layer
			UpdateInputWeightMatrix(i, inputHiddenWeights, hiddenLayerError, input);
		}
			
		//as long as the average difference is greater than the minimum allowed error 
		//we must continue iterating through training
		CalculateAverageDifference(aveDiff, sumDiff);
			
		if (aveDiff < 0)
		{
			aveDiff *= -1.0;
		}
		sumDiff = 0.0;
		
		epoch++;
		std::cout << "Iteration number: " << epoch << std::endl;

		if (aveDiff < minError && epoch >= 1000)
		{
			trainingFlag = false;
		}

		aveDiff = 0.0;
	}	//finish training
}

void TestingPhase(std::ifstream &testingFile, std::ofstream &outputFile, 
	std::vector< std::vector<float> > &input, std::vector<float> &hiddenNet, std::vector<float> &outputNet,
	std::vector<float> &hiddenActOutput, std::vector<float> &outputActOutput,
	std::vector< std::vector<float> > &inputHiddenWeights,
	std::vector< std::vector<float> > &hiddenOutputWeights, std::vector<int> &results)
{
	bool trainingFlag = true;
	float sumDiff = 0.0, aveDiff = 0.0;
	int epoch = 0, count = 0;

	//training 
	testingFile.open("alphabetTesting.txt");
	outputFile.open("alphabetOUT.txt");
	while (!testingFile.eof())		//reading in inputs and targets
	{
		ReadTestInputFromFile(testingFile, input);

		for (int i = 0; i < NUMBER_TEST_LETTERS; i++)	//for each letter in test file
		{
			//reset hidden node net values, output node net values, 
			//actual output of hidden and output layers, and error vectors to 0
			std::fill(hiddenNet.begin(), hiddenNet.end(), 0);
			std::fill(outputNet.begin(), outputNet.end(), 0);
			std::fill(hiddenActOutput.begin(), hiddenActOutput.end(), 0);
			std::fill(outputActOutput.begin(), outputActOutput.end(), 0);
			//calculate net values of the hidden layer nodes
			//net values are calculated by summing each input node
			//multipled by their corresponding weights in the weight matrix
			PropogateHiddenLayer(i, hiddenNet, input, inputHiddenWeights);		//summation of bias and inputs
																				//put hidden node net values through sigmoid activation function to get their output
																				//to get the output from the hidden nodes that are inputs to the outer layer node we must use the
																				//sigmoid function				
			ActivationFunction(hiddenActOutput, hiddenNet, HIDDEN_NODES);

			//calculate net value of outer layer node
			//this calculation uses the bias and the outputs from the two hidden layer nodes
			PropogateOutputLayer(outputNet, hiddenActOutput, hiddenOutputWeights);	//calculate output of outer layer node using sigmoid activation function
																					//using the net value just calculated we can use the sigmoid function on it to get our actual output 
																					//of the outer layer node
			ActivationFunction(outputActOutput, outputNet, OUTPUT_NODES);

			for (int i = 0; i < OUTPUT_NODES; i++)
			{
				if (outputActOutput[i] < 0.500)
				{
					results[i] = 0;
				}
				else
				{
					results[i] = 1;
				}
			}
			
			std::ostream_iterator<int> outputIterator(outputFile, " ");
			std::copy(results.begin(), results.end(), outputIterator);
			outputFile << std::endl;
		}

		std::cout << "Finished a testing Iteration..." << std::endl;

	} //finish testing

	testingFile.close();
	outputFile.close();
} 
