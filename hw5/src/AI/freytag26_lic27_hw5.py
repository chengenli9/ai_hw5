import random
import sys
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *

import random
import sys

sys.path.append("..")  # so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *
import numpy as np

## ------------- Neural Network ---------------

INPUT_NODES = 4
HIDDEN_NODES = 8
OUTPUT_NODES = 1
LEARNING_RATE = 0.5

## initalize random weights bewteen -1 and 1
np.random.seed(1)
hiddenLayerWeights = 2 * np.random.rand(INPUT_NODES + 1, HIDDEN_NODES) - 1  # 40 nodes
outputLayerWeights = 2 * np.random.rand(HIDDEN_NODES + 1, OUTPUT_NODES) - 1  # 9 nodes

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


# ---- output function ----
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

    return finalOutput, hiddenOutput

def backwardPass(inputs, hiddenLayerWeights, outputLayerWeights, target):
    # error calculation
    finalOutput, hiddenOutput = forwardPass(inputs, hiddenLayerWeights, outputLayerWeights)
    outputError = np.array(target - finalOutput)
    hiddenErrorTerms = np.dot(outputError, sigmoidDerivative(finalOutput))
    hiddenError = np.array(hiddenErrorTerms * hiddenLayerWeights)

    # adjusted weights
    adjustedOutputLayerWeights = []
    for i in range(outputLayerWeights):
        adjustedOutputLayerWeights.append(outputLayerWeights[i] + LEARNING_RATE * outputError * sigmoidDerivative(finalOutput) * hiddenOutput[i])

    adjustedHiddenLayerWeights = []
    for i in range(hiddenLayerWeights):
        adjustedHiddenLayerWeights.append(hiddenLayerWeights[i] + LEARNING_RATE * hiddenError * inputs[i%8])

    return np.array(adjustedOutputLayerWeights), np.array(adjustedHiddenLayerWeights)



# init training round
epoch = 0

# training loop
while True:
    epoch += 1
    errors = []

    # randomly pick 10 examples
    for i in range(10):
        inputs, target = random.choice(examples)
        inputs = np.array(inputs)  # convert inputs to np array
        target = np.array(target).reshape(1, -1)

        # forward pass
        finalOutput, hiddenOutput = forwardPass(inputs, hiddenLayerWeights, outputLayerWeights)
        adjustedOutputLayerWeights, adjustedHiddenLayerWeights = backwardPass(inputs, hiddenLayerWeights, outputLayerWeights, target)

        # ---- to be continued ----

    avgError = np.mean(errors)

    if avgError < 0.5:
        break


##
#AIPlayer
#Description: The responsbility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):

    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "Big Brain Bot")
    ##
    #getPlacement
    #
    #Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    #Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    #Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
    
    ##
    #getMove
    #Description: Gets the next move from the Player.
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: The Move to be made
    ##
    def getMove(self, currentState):
        moves = listAllLegalMoves(currentState)
        selectedMove = moves[random.randint(0,len(moves) - 1)];

        #don't do a build move if there are already 3+ ants
        numAnts = len(currentState.inventories[currentState.whoseTurn].ants)
        while (selectedMove.moveType == BUILD and numAnts >= 3):
            selectedMove = moves[random.randint(0,len(moves) - 1)];
            
        return selectedMove
    
    ##
    #getAttack
    #Description: Gets the attack to be made from the Player
    #
    #Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    #registerWin
    #
    # This agent doens't learn
    #
    def registerWin(self, hasWon):
        #method templaste, not implemented
        pass

    
