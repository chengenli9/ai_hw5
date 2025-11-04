import numpy as np
import random


INPUT_NODES = 4
HIDDEN_NODES = 8
OUTPUT_NODES = 1
LEARNING_RATE = 0.5

## initalize random weights bewteen -1 and 1
np.random.seed(1)
hiddenLayerWeights = 2 * np.random.rand(INPUT_NODES + 1, HIDDEN_NODES) - 1 # 40 nodes
outputLayerWeights = 2 * np.random.rand(HIDDEN_NODES + 1, OUTPUT_NODES) - 1 # 9 nodes


# training data
examples = [
    ([0, 0, 0, 0], [0]),
    ([0, 0, 0, 1], [1]),
    ([0, 0, 1, 0], [0]),
    ([0, 0, 1, 1], [1]),
    ([0, 1, 0, 0], [0]),
    ([0, 1, 0, 1], [1]),
    ([0, 1, 1, 0], [0]),
    ([0, 1, 1, 1], [1]),
    ([1, 0, 0, 0], [1]),
    ([1, 0, 0, 1], [1]),
    ([1, 0, 1, 0], [1]),
    ([1, 0, 1, 1], [1]),
    ([1, 1, 0, 0], [0]),
    ([1, 1, 0, 1], [0]),
    ([1, 1, 1, 0], [0]),
    ([1, 1, 1, 1], [1])
]

# activation function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoidDerivative(x):
    return x * (1.0 - x)


# forward propagation 
def forwardPass(inputs, hiddenLayerWeights, outputLayerWeights):
    # add bias to input
    inputWithBias = np.append(inputs, 1).reshape(1, -1)

    # hidden layer
    hiddenInput = np.dot(inputWithBias, hiddenLayerWeights)
    hiddenOutput = sigmoid(hiddenInput) 

    # add bias to hidden layer
    hiddenLayerWithBias = np.append(hiddenInput, 1).reshape(1, -1)

    # output layer
    finalInput = np.dot(hiddenLayerWithBias, outputLayerWeights)
    finalOutput = sigmoid(finalInput)

    # 8 hiddenOutput for every 1 finalOutput
    return finalOutput, hiddenOutput 


# source: https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/ and ChatGPT
# back propagation
def backwardPass(inputs, target, finalOutput, hiddenOutput,
                 hiddenLayerWeights, outputLayerWeights):
    # Convert to 2D arrays 
    inputs = inputs.reshape(1, -1)
    hiddenOutput = hiddenOutput.reshape(1, -1)
    target = np.array(target).reshape(1, -1)

    # Add bias to inputs and hidden layer
    inputWithBias = np.append(inputs, 1).reshape(1, -1)
    hiddenWithBias = np.append(hiddenOutput, 1).reshape(1, -1)

    # Calculate output error and delta 
    outputError = target - finalOutput
    outputDelta = outputError * sigmoidDerivative(finalOutput)

    # Calculate hidden layer error and delta 
    hiddenError = outputDelta.dot(outputLayerWeights[:-1].T) # exclude the bias
    hiddenDelta = hiddenError * sigmoidDerivative(hiddenOutput)

    # Update the weights 
    outputLayerWeights += LEARNING_RATE * hiddenWithBias.T.dot(outputDelta)
    hiddenLayerWeights += LEARNING_RATE * inputWithBias.T.dot(hiddenDelta)

    return hiddenLayerWeights, outputLayerWeights, abs(outputError[0][0])



# ----- Training the NN -----

# init training round
epoch = 0

# training loop
while True:
    epoch += 1
    errors = []

    # randomly pick 10 examples
    for i in range(10):
        inputs, target = random.choice(examples) 
        inputs = np.array(inputs) # convert inputs to np array
        target = np.array(target).reshape(1, -1)

        # forward pass
        finalOutput, hiddenOutput = forwardPass(inputs, hiddenLayerWeights, outputLayerWeights)

        # update the weights using backpropagation 
        hiddenLayerWeights, outputLayerWeights, error = backwardPass(
            inputs, target, finalOutput, hiddenOutput,
            hiddenLayerWeights, outputLayerWeights
        )
        # add to the list of errors
        errors.append(error) 

    averageError = np.mean(errors)
    print(f"Epoch {epoch}: average error = {averageError:.4f}")

    # stop the loop when average error is below 0.05
    if averageError < 0.05:
        break