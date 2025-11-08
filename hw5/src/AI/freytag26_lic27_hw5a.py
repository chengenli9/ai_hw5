# HW5a
# Authors: Chengen and Leonie

import numpy as np
import random
import sys
sys.path.append("..")
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *
# import matplotlib.pyplot as plt

# initalize how many nodes we want for the NN
INPUT_NODES = 4
HIDDEN_NODES = 8
OUTPUT_NODES = 1
LEARNING_RATE = 0.5

# initalize random weights bewteen -1 and 1
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
    hiddenLayerWithBias = np.append(hiddenInput, 1).reshape(1, -1)

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

# training loop
while True:
    epoch += 1
    errors = []

    # randomly pick 10 examples
    for i in range(10):
        inputs, target = random.choice(examples) 
        inputs = np.array(inputs).reshape(1, -1) # convert inputs to np array
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
    errorData.append(averageError)
    print(f"Epoch {epoch}: average error = {averageError:.4f}")


    # stop the loop when average error is below 0.05
    if averageError < 0.05:
        break


# # --- plot to visualize growth ---
# plt.plot(errorData)
# plt.title("Neural Network Learning Curve")
# plt.xlabel("Epoch")
# plt.ylabel("Average Error")
# plt.grid(True)
# plt.show()



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
        super(AIPlayer,self).__init__(inputPlayerId, "partA")
    
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
    #getMove
    #Description: Gets the next move from the AIPlayer
    #
    #
    #Return: The Move with the best evaluation 
    ##
    def getMove(self, currentState):
        moves = listAllLegalMoves(currentState)
        nodes = []
        
        for move in moves:
            nextState = getNextState(currentState, move)
            utility_score = self.utility(currentState, nextState)
            node = self.node(move, nextState, utility_score, None)
            nodes.append(node)
        
        best = self.bestMove(nodes)
        print((best["evaluation"] - 1.0) / 0.8)
        return best["move"]
    
    
    
    ##
    #getAttack
    #Description: Gets the attack to be made from the Player
    #
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        return enemyLocations[0]

    ##
    #registerWin
    #
    def registerWin(self, hasWon):
        pass


    ##
    # utility
    #
    # examines a GameState object and returns a heuristic guess of how "good" that game state is on a scale of 0..1.
    # Start of the game should return 0.5
    # When the game is almost won
    #
    def utility(self, currentState, nextState):
        myId = currentState.whoseTurn
        next_inv = nextState.inventories[myId]
        
        # Evaluate absolute position in next state (0-1 scale)
        
        # 1. Food progress (0-1, where 1 = food goal reached)
        food_score = min(next_inv.foodCount / float(FOOD_GOAL), 1.0)
        
        # 2. Route efficiency (normalize to 0-1)
        route_score = self.compute_route_score(nextState)
        # Assuming route score needs normalization - adjust based on your range
        normalized_route = max(0, min(1, (route_score + 1) / 2))  # if route_score is [-1,1]
        
        # 3. Unit composition (normalize to 0-1) 
        unit_score = self.compute_unit_composition_score(nextState)
        # Normalize based on your unit score range
        normalized_units = max(0, min(1, unit_score))  # assuming already 0-1
        
        # 4. Military aggression (normalize to 0-1)
        aggro_score = self.compute_rsoldier_aggression_score(nextState)
        # Normalize based on your aggro score range  
        normalized_aggro = max(0, min(1, aggro_score))  # assuming already 0-1
        
        # Weights (should sum to 1.0 for proper scaling)
        food_w = 0.40      # Food is most important for winning
        route_w = 0.20     # Efficiency matters
        units_w = 0.25     # Unit composition important
        aggro_w = 0.15     # Military presence
        
        # Weighted combination (results in 0-1 scale)
        base_utility = (food_w * food_score + 
                    route_w * normalized_route + 
                    units_w * normalized_units + 
                    aggro_w * normalized_aggro)
        
        # Win condition check - if food goal reached, should be close to 1.0
        if next_inv.foodCount >= FOOD_GOAL:
            base_utility = max(base_utility, 0.95)  # Near-certain win
        
        # Loss condition check (optional - if you have lose conditions)
        # Example: if critically low on food
        if next_inv.foodCount == 0:
            base_utility = min(base_utility, 0.05)  # Near-certain loss
        
        # Ensure strictly within [0,1] bounds
        return base_utility
        
        
        
    

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
            "evaluation": (utility + depth),
            "parent": parent
        }
    
    ## Best move from a list of nodes
    #
    def bestMove(self, nodes):
        return max(nodes, key=lambda x: x["evaluation"])
    

# clamp function for capping min and max values
def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))
