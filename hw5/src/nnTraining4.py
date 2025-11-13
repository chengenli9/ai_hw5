# HW5a
# Authors: Chengen and Leonie

import numpy as np
import random
import matplotlib.pyplot as plt

# initalize how many nodes we want for the NN
INPUT_NODES = 14
HIDDEN_NODES = 20
OUTPUT_NODES = 1
LEARNING_RATE = 0.05

# # initalize random weights bewteen 0 and 1
# np.random.seed(1)
# hiddenLayerWeights = np.random.rand(INPUT_NODES + 1, HIDDEN_NODES) 
# outputLayerWeights = np.random.rand(HIDDEN_NODES + 1, OUTPUT_NODES) 
# Xavier/Glorot initialization instead of uniform [0,1]
hiddenLayerWeights = np.random.randn(INPUT_NODES + 1, HIDDEN_NODES) * np.sqrt(2.0 / (INPUT_NODES + HIDDEN_NODES))
outputLayerWeights = np.random.randn(HIDDEN_NODES + 1, OUTPUT_NODES) * np.sqrt(2.0 / (HIDDEN_NODES + OUTPUT_NODES))


# training data
# examples = [
#     ([0, 0, 0, 0], [0]),
#     ([0, 0, 0, 1], [1]),
#     ([0, 0, 1, 0], [0]),
#     ([0, 0, 1, 1], [1]),
#     ([0, 1, 0, 0], [0]),
#     ([0, 1, 0, 1], [1]),
#     ([0, 1, 1, 0], [0]),
#     ([0, 1, 1, 1], [1]),
#     ([1, 0, 0, 0], [1]),
#     ([1, 0, 0, 1], [1]),
#     ([1, 0, 1, 0], [1]),
#     ([1, 0, 1, 1], [1]),
#     ([1, 1, 0, 0], [0]),
#     ([1, 1, 0, 1], [0]),
#     ([1, 1, 1, 0], [0]),
#     ([1, 1, 1, 1], [1])
# ]

# load training data from npz file
data = np.load("training_data.npz")
features = data["features"]

# split into X (inputs) Y (outputs)
X = features[:, :-1] # first 14
Y = features[:, -1:] # last column as target

# activation function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoidDerivative(x):
    return x * (1.0 - x)


## forwardPass
# Description: returns the output of the output along with the inputs from the hidden nodes
#
def forwardPass(inputs, hiddenLayerWeights, outputLayerWeights):
    # add bias to input
    inputWithBias = np.append(inputs, 1).reshape(1, -1)

    # hidden layer
    hiddenInput = np.dot(inputWithBias, hiddenLayerWeights)
    hiddenOutput = sigmoid(hiddenInput) 

    # add bias to hidden layer
    hiddenLayerWithBias = np.append(hiddenOutput, 1).reshape(1, -1)

    # output layer
    finalInput = np.dot(hiddenLayerWithBias, outputLayerWeights)
    finalOutput = sigmoid(finalInput)

    # 8 hiddenOutput for every 1 finalOutput
    return finalOutput, hiddenOutput 



## backwardPass
# Description: returns the updated weight values for hiddenlayer and outputlayers. 
#               Also returns the output error after one iteration
# 
# source: https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/ and ChatGPT
#
def backwardPass(inputs, target, finalOutput, hiddenOutput,
                 hiddenLayerWeights, outputLayerWeights):
    
    hiddenOutput = hiddenOutput.reshape(1, -1)

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
errorData = []
accuracyData = []

# training loop
while True:
    epoch += 1
    errors = []

    INITIAL_LR = 0.1
    # In training loop:
    LEARNING_RATE = INITIAL_LR / (1 + epoch * 0.001)

    # randomly pick 10 examples
    for i in range(200):
        idx = random.randint(0, len(X) - 1)
        inputs = X[idx]
        target = Y[idx]

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
    errorData.append(averageError)
    print(f"Epoch {epoch}: average error = {averageError:.4f}")


    # stop the loop when average error is below 0.05
    if averageError < 0.02 or epoch >= 10000:
        print("training complete!")

        break
    



# --- save the weights to a npz file ---
np.savez('trained_weights.npz', 
         hidden_weights=hiddenLayerWeights, 
         output_weights=outputLayerWeights)
print("Weights saved to 'trained_weights.npz'")

# --- plot to visualize growth ---
plt.plot(errorData)
plt.title("Neural Network Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Average Error")
plt.grid(True)
plt.show()

