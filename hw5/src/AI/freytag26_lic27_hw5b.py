# HW5a
# Authors: Chengen and Leonie

import numpy as np
import random
import sys
import os
import matplotlib.pyplot as plt

sys.path.append("..")
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *

# make sure load and save files in the same directory 
AI_DIR = os.path.dirname(os.path.abspath(__file__))



# # initalize how many nodes we want for the NN
INPUT_NODES = 14
#NUM_LAYERS = 2
HIDDEN_NODES = 10
OUTPUT_NODES = 1
LEARNING_RATE = 0.1

# np.random.seed(1)
# hiddenLayerWeights = np.random.rand(INPUT_NODES + 1, HIDDEN_NODES) 
# outputLayerWeights = np.random.rand(HIDDEN_NODES + 1, OUTPUT_NODES) 

# load the weights
weights_file = os.path.join(AI_DIR, "trained_weights.npz")
if os.path.exists(weights_file):
    weights = np.load(weights_file)
    hiddenLayerWeights = weights['hidden_weights']
    outputLayerWeights = weights['output_weights']
    print(f"Loaded weights from {weights_file}")
else:
    hiddenLayerWeights = np.random.rand(INPUT_NODES + 1, HIDDEN_NODES) 
    outputLayerWeights = np.random.rand(HIDDEN_NODES + 1, OUTPUT_NODES)
    np.savez(weights_file, 
             hidden_weights=hiddenLayerWeights,
             output_weights=outputLayerWeights)
    print(f"Initialized new weights in {weights_file}")

# Load training data
data_file = os.path.join(AI_DIR, "training_data.npz")
if os.path.exists(data_file):
    data = np.load(data_file)
    features = data["features"]
    print(f"Loaded training data from {data_file}")
else:
    features = np.array([]).reshape(0, INPUT_NODES + 1)
    np.savez(data_file, features=features)
    print(f"Created new training data file: {data_file}")



# # activation function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoidDerivative(x):
    return x * (1.0 - x)


# ## forwardPass
# # Description: returns the output of the output along with the inputs from the hidden nodes
# #
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





# ## backwardPass
# # Description: returns the updated weight values for hiddenlayer and outputlayers. 
# #               Also returns the output error after one iteration
# # 
# # source: https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/ and ChatGPT
# #
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



def trainNN():
    global hiddenLayerWeights, outputLayerWeights

    # Reload training data collected during this game
    data_file = os.path.join(AI_DIR, "training_data.npz")
    if not os.path.exists(data_file):
        print("No training data found!")
        return
    
    data = np.load(data_file)
    features = data["features"]
    
    # Check if we have data
    if len(features) == 0:
        print("No training data collected this game!")
        return

    # # split into X (inputs) Y (outputs)
    X = features[:, :-1] # first 14
    Y = features[:, -1:] # last column as target


    epoch = 0
    errorData = []


    # training loop
    while True:
        epoch += 1
        errors = []

        # shuffles all rows each epoch
        indices = np.random.permutation(len(X) - 30)

        # randomly pick 10 examples
        for idx in indices:
            #idx = random.randint(0, len(X) - 1)
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
        if averageError < 0.00001 or epoch >= 10000:
            print("training complete!")

            break
        

    weights_file = os.path.join(AI_DIR, "trained_weights.npz")

    # --- save the weights to a npz file ---
    np.savez(weights_file, 
            hidden_weights=hiddenLayerWeights, 
            output_weights=outputLayerWeights)
    print("Weights saved to 'trained_weights.npz'")

    



##
# AIPlayer
# Description: The responsbility of this class is to interact with the game by
# deciding a valid move based on a given game state. This class has methods that
# will be implemented by students in Dr. Nuxoll's AI course.
#
# Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):


    # __init__
    # Description: Creates a new Player
    #
    # Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer, self).__init__(inputPlayerId, "NN")

        self.epoch = 0
        # self.hiddenLayerWeights, self.outputLayerWeights = self.loadData() # ignore for now

        self.isTraining = True

    ##
    # getPlacement
    #
    # Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    # Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    # Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        if currentState.phase == SETUP_PHASE_1:
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    x = random.randint(0, 9)
                    y = random.randint(0, 3)
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    x = random.randint(0, 9)
                    y = random.randint(6, 9)
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]

    ##
    # getMove
    # Description: Gets the next move from the AIPlayer
    #
    #
    # Return: The Move with the best evaluation
    ##
    def getMove(self, currentState):
        # moves = listAllLegalMoves(currentState)
        # nodes = []

        # for move in moves:
        #     nextState = getNextState(currentState, move)
        #     utility_score = (self.utility(currentState, nextState) / 0.92) if self.isTraining else self.neuralNetworkUtility(nextState)
        #     # self.trainNN(currentState, utility_score)
        #     node = self.node(move, nextState, utility_score, None)
        #     nodes.append(node)

        # best = self.bestMove(nodes)
        moves = listAllLegalMoves(currentState)
        nodes = []
        
        # Current state scores
        cur_route_score = self.compute_route_score(currentState)
        myId = currentState.whoseTurn
        cur_food_frac = currentState.inventories[myId].foodCount / float(FOOD_GOAL)
        cur_unit_score = self.compute_unit_composition_score(currentState)
        cur_aggro_score = self.compute_rsoldier_aggression_score(currentState)
        
        # Weights
        route_w = 0.10
        food_w = 0.20
        units_w = 0.30
        aggro_w = 0.40

        for move in moves:
            nextState = getNextState(currentState, move)
            next_route_score = self.compute_route_score(nextState)
            next_food_frac = nextState.inventories[myId].foodCount / float(FOOD_GOAL)
            
            # Route delta
            route_term = next_route_score - cur_route_score
            if route_term > 0.25:
                route_term = 0.25
            elif route_term < -0.25:
                route_term = -0.25
            
            delta = route_w * route_term + food_w * (next_food_frac - cur_food_frac)

            # Units delta
            next_unit_score = self.compute_unit_composition_score(nextState)
            delta += units_w * (next_unit_score - cur_unit_score)
            
            # Aggression delta
            next_aggro_score = self.compute_rsoldier_aggression_score(nextState)
            delta += aggro_w * (next_aggro_score - cur_aggro_score)
            
            # Food completion bonus
            cur_food = currentState.inventories[myId].foodCount
            next_food = nextState.inventories[myId].foodCount
            food_gain = next_food - cur_food
            if food_gain > 0:
                delta += 0.10 * food_gain
            
            node = self.node(move, nextState, delta, None)
            nodes.append(node)
        
        best = self.bestMove(nodes)
        # print(self.utility(currentState) / 0.8)
        # return best["move"]

        normalUtility = (best['evaluation']) 
        data = self.getFeatures(currentState)
        data.append(normalUtility)
        #print(f"{data}")

        # Get the directory where this AI file is located
        ai_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(ai_dir, "training_data.npz")
        features = np.array(data)

        # Try to load existing data
        if os.path.exists(filename):
            data = np.load(filename, allow_pickle=True)
            featureData = data["features"]
            data.close()  # close file handle to avoid warning
            # Append a new row
            featureData = np.vstack((featureData, features))
        else:
            # First time: start with 1 row
            featureData = np.array([features])

        # Save updated data back
        np.savez(filename, features=featureData)

        return best["move"]
    

    ##
    # getAttack
    # Description: Gets the attack to be made from the Player
    #
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        return enemyLocations[0]

    ##
    # registerWin
    #
    def registerWin(self, hasWon):
        print("Game over! Training neural network...")
    
        # Train the NN with collected data from this game
        if self.isTraining == True:
            trainNN()
            
            # Clear the training data file for next game
            ai_dir = os.path.dirname(os.path.abspath(__file__))
            data_file = os.path.join(ai_dir, "training_data.npz")
            features = np.array([]).reshape(0, INPUT_NODES + 1)
            np.savez(data_file, features=features)
            print(f"Cleared training data for next game")
        
        return hasWon


    



    # ---------- Neural Network Stuff ------------

    # # count epochs
    # epoch = 0

    # activation function
    # def sigmoid(self, x):
    #     return 1.0 / (1.0 + np.exp(-x))

    # def sigmoidDerivative(self, x):
    #     return x * (1.0 - x)
    

    ## trainNN
    # Description: calls all methods for Neural Network training
    #
    # def trainNN(self, currentState, target):
    #     # init training round
    #     errors = []
    #     self.epoch += 1

    #     # inputs are the gameState features
    #     inputs = np.array(self.getFeatures(currentState))

    #     # forward pass
    #     finalOutput, hiddenOutput = self.forwardPass(inputs, self.hiddenLayerWeights, self.outputLayerWeights)

    #     # update the weights using backpropagation
    #     self.hiddenLayerWeights, self.outputLayerWeights, error = self.backwardPass(
    #         inputs, target, finalOutput, hiddenOutput,
    #         self.hiddenLayerWeights, self.outputLayerWeights
    #     )
    #     # add to the list of errors
    #     errors.append(error)

    #     averageError = np.mean(errors)
    #     print(f"Epoch {self.epoch}: average error = {averageError:.4f}")
        
       


    ## utility function for after NN training is finished
    # we get our final weights
    # 
    def neuralNetworkUtility(self, currentState):
        # load hiddenLayerWeights and outputLayerWeights from "trained_weights.npz"
        try:
            ai_dir = os.path.dirname(os.path.abspath(__file__))

            filename = os.path.join(ai_dir, "trained_weights.npz")
            weights = np.load(filename)
            hiddenLayerWeights = weights['hidden_weights']
            outputLayerWeights = weights['output_weights']
        except FileNotFoundError:
            raise FileNotFoundError("trained_weights.npz not found.")

        inputs = self.getFeatures(currentState)
        inputs = np.array(inputs)

        
        utility, _ = forwardPass(inputs, hiddenLayerWeights, outputLayerWeights)
        print(utility[0][0].item())

        return utility[0][0].item() # convert back to regular float





    # get input features
    def getFeatures(self, currentState):
        me = currentState.whoseTurn
        them = 1 - me

        ## comparable features (possibly assigning 0 if <0, 0.5 if = 0, 1 if > 0)

        # food difference between agent and opponent
        myFood = currentState.inventories[me].foodCount
        theirFood = currentState.inventories[them].foodCount
        food_difference = myFood - theirFood

        # health difference between agent’s and opponent’s queen and anthill
        health_difference = 0

        theirAnthill = currentState.inventories[them].getAnthill()
        myAnthill = currentState.inventories[me].getAnthill()
        theirQueen = None
        if len(getAntList(currentState, them, (QUEEN,))) > 0 and len(getAntList(currentState, me, (QUEEN,))) > 0:
            myQueen = getAntList(currentState, me, (QUEEN,))[0]
            myQueensHealth = myQueen.health
            theirQueen = getAntList(currentState, them, (QUEEN,))[0]
            theirQueensHealth = theirQueen.health
            myAnthillHealth = myAnthill.captureHealth
            theirAnthillHealth = theirAnthill.captureHealth
            health_difference = (myQueensHealth + myAnthillHealth * 3) - (theirQueensHealth + theirAnthillHealth * 3)
        else:
            return [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # soldier number difference between agent and opponent
        mySoldiers = getAntList(currentState, me, (SOLDIER,))
        theirSoldiers = getAntList(currentState, them, (SOLDIER,))
        soldier_difference = len(mySoldiers) - len(theirSoldiers)

        # drone number difference between agent and opponent
        myDrones = getAntList(currentState, me, (DRONE,))
        theirDrones = getAntList(currentState, them, (DRONE,))
        drone_difference = len(myDrones) - len(theirDrones)

        # ranged soldier number difference between agent and opponent
        myRSoldiers = getAntList(currentState, me, (R_SOLDIER,))
        theirRSoldiers = getAntList(currentState, them, (R_SOLDIER,))
        r_soldier_difference = len(myRSoldiers) - len(theirRSoldiers)

        # difference in offensive capability (1 if >0, 0 if <= 0)
        myAttAnts = getAntList(currentState, me, (DRONE, SOLDIER, R_SOLDIER))
        theirAttAnts = getAntList(currentState, them, (DRONE, SOLDIER, R_SOLDIER))
        offensive_difference = len(myAttAnts) - len(theirAttAnts)
        offensive_capability = 1 if offensive_difference > 0 else 0

        ## distance features (setting min max limits individually)

        # average distance between the agent’s worker and food
        worker_food_distance = 0
        myWorkers = getAntList(currentState, me, (WORKER,))
        myFoodObjs = getConstrList(currentState, me, (FOOD,))
        myTunnel = currentState.inventories[me].getTunnels()[0]
        distances = []
        carryingWorkers = 0
        if myWorkers and myAnthill and myFoodObjs:
            for worker in myWorkers:
                # goal for carrying workers is the tunnel/anthill to drop it
                if worker.carrying:
                    carryingWorkers += 1
                    distance1 = approxDist(myAnthill.coords, worker.coords)
                    distance2 = approxDist(myTunnel.coords, worker.coords)
                    if worker.coords == myAnthill.coords or worker.coords == myTunnel.coords:
                        distance1 = 4
                # goal for workers not carrying food is one of the two food objects to get it
                else:
                    distance1 = approxDist(myFoodObjs[0].coords, worker.coords)
                    distance2 = approxDist(myFoodObjs[1].coords, worker.coords)
                    if worker.coords == myFoodObjs[0].coords or worker.coords == myFoodObjs[1].coords:
                        distance1 = 4
                # chooses the shorter distance of the two goal objects
                distances.append(distance1) if distance1 < distance2 else distances.append(distance2)
            # add some emphasis on carrying food to incentivise picking up food
            worker_food_distance = (sum(distances) / len(distances)) - (carryingWorkers * 0.5) if len(
                distances) > 0 else 0

        # average distance between opponent’s offensive ants and agent’s queen
        their_offense_queen_distance = 0
        distances = []
        if theirAttAnts and myQueen:
            for ant in theirAttAnts:
                distances.append(approxDist(ant.coords, myQueen.coords))
            their_offense_queen_distance = (sum(distances) / len(distances))

        # average distance between agent’s offense and opponent’s anthill
        my_offense_anthill_distance = 0
        distances = []
        if myAttAnts and theirAnthill:
            for ant in myAttAnts:
                distances.append(approxDist(ant.coords, theirAnthill.coords))
            my_offense_anthill_distance = (sum(distances) / len(distances))

        # average distance between opponent’s offense and agent’s anthill
        their_offense_anthill_distance = 0
        distances = []
        if theirAttAnts and myAnthill:
            for ant in theirAttAnts:
                distances.append(approxDist(ant.coords, myAnthill.coords))
            their_offense_anthill_distance = (sum(distances) / len(distances))

        # average distance between agent’s offense and opponent’s queen
        distances = []
        my_offense_queen_distance = 0
        if myAttAnts and theirQueen:
            for ant in myAttAnts:
                distances.append(approxDist(ant.coords, theirQueen.coords))
            my_offense_queen_distance = (sum(distances) / len(distances))

        # average dist between agent’s offense and opponent’s offense closest to queen
        my_defense_offense_distance = 0
        if theirAttAnts and myAttAnts:
            closestAnt = theirAttAnts[0]
            for ant in theirAttAnts:
                if approxDist(closestAnt.coords, myQueen.coords) > approxDist(ant.coords, myQueen.coords):
                    closestAnt = ant
            distances = []
            for ant in myAttAnts:
                distances.append(approxDist(ant.coords, closestAnt.coords))
            my_defense_offense_distance = (sum(distances) / len(distances))

        ## irrelevant features

        # average distance from workers to queen
        distances = []
        my_workers_queen_distance = 0
        for worker in myWorkers:
            distances.append(approxDist(worker.coords, myQueen.coords))
        if len(distances) > 0:
            my_workers_queen_distance = (sum(distances) / len(distances))

        # distance between agent’s and opponent’s queen
        # not reversing (for seeing the difference in the method + more distance better here)
        queens_distance = approxDist(myQueen.coords, theirQueen.coords)

        # add features' values to a list
        features = []
        features.append(food_difference)
        features.append(health_difference)
        features.append(soldier_difference)
        features.append(drone_difference)
        features.append(r_soldier_difference)
        features.append(offensive_capability)
        features.append(worker_food_distance)
        features.append(their_offense_queen_distance)
        features.append(my_offense_anthill_distance)
        features.append(their_offense_anthill_distance)
        features.append(my_offense_queen_distance)
        features.append(my_defense_offense_distance)
        features.append(my_workers_queen_distance)
        features.append(queens_distance)

        # limit the feature values to floats between 0 and 1
        featureInputs = []
        for feature in features:
            match feature:
                case f if f < 0:
                    featureInputs.append(0)
                case f if f > 10:
                    featureInputs.append(1)
                case f if f == 0:
                    featureInputs.append(0.05)
                case _:
                    featureInputs.append(feature / 10.0)

        # return adjusted feature values for NN
        return featureInputs

    # ##
    # # utility
    # #
    # # examines a GameState object and returns a heuristic guess of how "good" that game state is on a scale of 0..1.
    # # Start of the game should return 0.5
    # # When the game is almost won
    # #
    # def utility(self, currentState, nextState):
    #     myId = currentState.whoseTurn
    #     next_inv = nextState.inventories[myId]

    #     # Evaluate absolute position in next state (0-1 scale)

    #     # 1. Food progress (0-1, where 1 = food goal reached)
    #     food_score = min(next_inv.foodCount / float(FOOD_GOAL), 1.0)

    #     # 2. Route efficiency (normalize to 0-1)
    #     route_score = self.compute_route_score(nextState)
    #     # Assuming route score needs normalization - adjust based on your range
    #     normalized_route = max(0, min(1, (route_score + 1) / 2))  # if route_score is [-1,1]

    #     # 3. Unit composition (normalize to 0-1)
    #     unit_score = self.compute_unit_composition_score(nextState)
    #     # Normalize based on your unit score range
    #     normalized_units = max(0, min(1, unit_score))  # assuming already 0-1

    #     # 4. Military aggression (normalize to 0-1)
    #     aggro_score = self.compute_rsoldier_aggression_score(nextState)
    #     # Normalize based on your aggro score range
    #     normalized_aggro = max(0, min(1, aggro_score))  # assuming already 0-1

    #     # Weights (should sum to 1.0 for proper scaling)
    #     food_w = 0.40  # Food is most important for winning
    #     route_w = 0.20  # Efficiency matters
    #     units_w = 0.25  # Unit composition important
    #     aggro_w = 0.15  # Military presence


    #     # Weighted combination (results in 0-1 scale)
    #     base_utility = (food_w * food_score +
    #                     route_w * normalized_route +
    #                     units_w * normalized_units +
    #                     aggro_w * normalized_aggro)
        
    #     # print(f"{route_w} {food_w} {units_w} {aggro_score}")

    #     # Win condition check - if food goal reached, should be close to 1.0
    #     if next_inv.foodCount >= FOOD_GOAL:
    #         base_utility = max(base_utility, 0.95)  # Near-certain win

    #     # Loss condition check (optional - if you have lose conditions)
    #     # Example: if critically low on food
    #     if next_inv.foodCount == 0:
    #         base_utility = min(base_utility, 0.05)  # Near-certain loss

    #     # Ensure strictly within [0,1] bounds
    #     return base_utility


    def utility(self, currentState): 
        route_score = self.compute_route_score(currentState)         
        myId = currentState.whoseTurn         
        enemyId = 1 - myId
        myInv = currentState.inventories[myId]
        enemyInv = currentState.inventories[enemyId]
        
        # Calculate relative scores (compare to enemy)
        my_food = myInv.foodCount
        enemy_food = enemyInv.foodCount
        
        # Food score: relative to enemy, starting at 0.5
        if my_food + enemy_food == 0:
            food_score = 0.5  # Neutral at game start
        else:
            food_score = 0.5 + (my_food - enemy_food) / (2.0 * float(FOOD_GOAL))
            food_score = max(0.0, min(1.0, food_score))
        
        # Route score: already normalized, but ensure it starts neutral
        if route_score == 0.0:  # No workers yet
            route_score = 0.5
        
        # Unit composition: relative to ideal composition
        unit_comp = self.compute_unit_composition_score(currentState)
        # Convert from [0,1] to centered around 0.5
        # Perfect composition (0.5 worker + 0.5 rsoldier) = 1.0 becomes 0.75
        # No units = 0.0 -> becomes 0.25
        # Start of game (just worker) = 0.5 becomes 0.5
        unit_comp_centered = 0.25 + (unit_comp * 0.5)
    
        # Aggression score: relative effectiveness
        my_aggro = self.compute_rsoldier_aggression_score(currentState)
        
        # Enemy aggression (defensive consideration)
        enemy_rsoldiers = getAntList(currentState, enemyId, (R_SOLDIER,))
        my_workers = getAntList(currentState, myId, (WORKER,))
        my_anthill = myInv.getAnthill()
        
        enemy_threat = 0.0
        if len(enemy_rsoldiers) > 0 and my_anthill:
            total_threat = 0.0
            for enemy_rs in enemy_rsoldiers:
                # Threat based on how close enemy soldiers are to our assets
                threats = []
                for worker in my_workers:
                    dist = stepsToReach(currentState, enemy_rs.coords, worker.coords)
                    if dist >= 0:
                        threats.append(dist)
                
                hill_dist = stepsToReach(currentState, enemy_rs.coords, my_anthill.coords)
                if hill_dist >= 0:
                    threats.append(hill_dist)
                
                if threats:
                    min_threat_dist = min(threats)
                    threat_score = max(0.0, 1.0 - (min_threat_dist / 15.0))
                    total_threat += threat_score
            
            enemy_threat = min(1.0, total_threat / len(enemy_rsoldiers))
        
        # Net aggression: our offense minus their threat
        net_aggro = 0.5 + (my_aggro - enemy_threat) * 0.5
        net_aggro = max(0.0, min(1.0, net_aggro))
        
        # Game phase adaptive weights
        total_food = my_food + enemy_food
        
        if total_food <= 3: # Early game
            route_w = 0.4
            food_w = 0.4  
            units_w = 0.2
            aggro_w = 0.0
        elif total_food <= 8: # Mid game
            route_w = 0.3
            food_w = 0.3
            units_w = 0.25
            aggro_w = 0.15
        else:  # Late game
            route_w = 0.2
            food_w = 0.35
            units_w = 0.2
            aggro_w = 0.25
        
        # Win/lose conditions override
        if my_food >= FOOD_GOAL:
            return 1.0  # We won
        elif enemy_food >= FOOD_GOAL:
            return 0.0  # We lost
        
        # Calculate weighted score
        score = (route_w * route_score + 
                food_w * food_score + 
                units_w * unit_comp_centered + 
                aggro_w * net_aggro)
        
        # Ensure score stays in bounds and starts near 0.5
        return max(0.0, min(1.0, score))

    

    ## compute_unit_composition_score
    # computes a score [0,1] based on having a balanced set of unit types
    # Higher score means more balanced
    def compute_unit_composition_score(self, state):
        myId = state.whoseTurn
        myInv = state.inventories[myId]

        has_worker = any(a.type == WORKER for a in myInv.ants)
        has_r_soldier = any(a.type == R_SOLDIER for a in myInv.ants)

        score = 0.0
        if has_worker:
            score += 0.5
        if has_r_soldier:
            score += 0.5
        return score

    ##
    # compute_route_score
    #
    # computes a score [0,1] based on the delivery potential of current workers
    # Higher score means shorter routes on average
    #
    def compute_route_score(self, state):
        myId = state.whoseTurn
        myInv = state.inventories[myId]
        myWorkers = getAntList(state, myInv.player, (WORKER,))
        foods = getConstrList(state, NEUTRAL, (FOOD,))

        deposit_coords = []
        if myInv.getAnthill() is not None:
            deposit_coords.append(myInv.getAnthill().coords)
        for t in myInv.getTunnels():
            deposit_coords.append(t.coords)

        # helper function to find the shortest distance from a coord to any deposit
        def min_deposit_dist_from(coord):
            best = None
            for dep in deposit_coords:
                d = stepsToReach(state, coord, dep)
                if d >= 0:
                    best = d if best is None else min(best, d)
            return best

        MAX_ROUTE = 20.0
        route_lengths = []

        for w in myWorkers:
            if getattr(w, 'carrying', False):
                ddep = min_deposit_dist_from(w.coords)
                if ddep is not None:
                    route_lengths.append(ddep)
            else:
                best_dtofood = None
                best_food = None
                for f in foods:
                    dtof = stepsToReach(state, w.coords, f.coords)
                    if dtof >= 0 and (best_dtofood is None or dtof < best_dtofood):
                        best_dtofood = dtof
                        best_food = f
                if best_food is not None and best_dtofood is not None:
                    ddep = min_deposit_dist_from(best_food.coords)
                    if ddep is not None:
                        route_lengths.append(best_dtofood + ddep)

        if len(route_lengths) == 0:
            return 0.0

        avg_route = sum(route_lengths) / float(len(route_lengths))
        route_score = 1.0 - clamp(avg_route / MAX_ROUTE)
        return route_score

    ## compute_rsoldier_aggression_score
    #
    #
    # makes a score [0.0, 1.0] based on how effectively r_soldiers are positioned to attack enemies
    # Priority: Enemy workers first, then enemy anthill/queen
    def compute_rsoldier_aggression_score(self, state):
        myId = state.whoseTurn
        enemyId = 1 - myId
        myInv = state.inventories[myId]
        enInv = state.inventories[enemyId]
        my_rsoldiers = [a for a in myInv.ants if a.type == R_SOLDIER]
        enemy_workers = getAntList(state, enemyId, (WORKER,))
        enemy_hill = enInv.getAnthill() if enInv is not None else None
        enemy_queen = enInv.getQueen() if enInv is not None else None

        if len(my_rsoldiers) == 0:
            return 0.0

        total_aggression_score = 0.0

        for rsoldier in my_rsoldiers:
            soldier_score = 0.0

            # PRIORITY 1: Target enemy workers (higher weight)
            if len(enemy_workers) > 0:
                best_worker_distance = float('inf')
                for worker in enemy_workers:
                    distance = stepsToReach(state, rsoldier.coords, worker.coords)
                    if distance >= 0:
                        best_worker_distance = min(best_worker_distance, distance)

                if best_worker_distance != float('inf'):
                    MAX_WORKER_DISTANCE = 10.0  # Smaller max for worker targeting
                    worker_score = 1.0 - clamp(best_worker_distance / MAX_WORKER_DISTANCE)
                    soldier_score += 0.7 * worker_score  # 70% weight for worker targeting

            # PRIORITY 2: Target enemy anthill/queen (lower weight, only if no workers or as secondary)
            secondary_targets = []
            if enemy_hill:
                secondary_targets.append(enemy_hill.coords)
            if enemy_queen:
                secondary_targets.append(enemy_queen.coords)

            if secondary_targets:
                best_secondary_distance = float('inf')
                for target_coords in secondary_targets:
                    distance = stepsToReach(state, rsoldier.coords, target_coords)
                    if distance >= 0:
                        best_secondary_distance = min(best_secondary_distance, distance)

                if best_secondary_distance != float('inf'):
                    MAX_SECONDARY_DISTANCE = 15.0
                    secondary_score = 1.0 - clamp(best_secondary_distance / MAX_SECONDARY_DISTANCE)

                    # If no enemy workers exist, give full weight to secondary targets
                    # Otherwise, give reduced weight (30%)
                    weight = 1.0 if len(enemy_workers) == 0 else 0.3
                    soldier_score += weight * secondary_score

            total_aggression_score += soldier_score

        # Average aggression score across all r_soldiers
        avg_aggression = total_aggression_score / len(my_rsoldiers)

        # Bonus for eliminating enemy workers (strategic progress)
        initial_enemy_workers = 2  # Assume enemy starts with ~2 workers typically
        if len(enemy_workers) < initial_enemy_workers:
            elimination_bonus = 0.5 * (initial_enemy_workers - len(enemy_workers))
            avg_aggression += elimination_bonus

        return clamp(avg_aggression)
    



    ## Node representation
    #
    def node(self, move, state, utility, parent, depth=1):
        return {
            "move": move,
            "state": state,
            "evaluation": (utility + 1),
            "parent": parent
        }

    ## Best move from a list of nodes
    #
    def bestMove(self, nodes):
        return max(nodes, key=lambda x: x["evaluation"])




# clamp function for capping min and max values
def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

