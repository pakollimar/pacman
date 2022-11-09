# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPosition).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPosition = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** CS5368 YOUR CODE HERE ***"
        "Decribe your function:"
        #Checking for the closest food near to the pacman and compute the distance between the position of the pacman to the food using manhattandistance
        #Finding the closest ghosts postions to the pacman
        #we compute the score accord to the nearest food and the ghost positions
        
        if action == 'Stop':
            return -1000000
        #get closest food for the pacman to eat
        closestFood = None
        closestFoodDistance = float('inf')
        for food in newFood.asList():
            foodDistance = manhattanDistance(food, newPosition)
            if foodDistance < closestFoodDistance:
                closestFood = food
                closestFoodDistance = foodDistance

        t = 0
        if closestFood:
            minimumDistance = manhattanDistance(newPosition, closestFood)
            t -= minimumDistance * .25

        # information about the ghost positions 
        ghostPosition = []
        for ghostState in newGhostStates:
            g = ghostState.configuration.pos
            ghostPosition.append(g)

        cGhost = None
        cGhostDist = float('inf')
        for g in ghostPosition:
            dist = manhattanDistance(newPosition, g)
            if dist < cGhostDist:
                cGhostDist = dist
                cGhost = g
        if cGhostDist <= 3:
            t -= (3 - cGhostDist) * 1000

        t += successorGameState.data.score

        if newPosition == currentGameState.getPacmanPosition():
            t -= 1
        return t

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def maxFun(self, gameState, numofGhosts, plyCount):
        eva = []
        if (plyCount == 0 or gameState.isLose() or gameState.isWin()):
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions():
            eva.append(self.minFun(gameState.generateSuccessor(self.index, action), numofGhosts, plyCount))

        return max(eva)

    def minFun(self, gameState, numofGhosts, plyCount):
        eva = []

        if gameState.isWin() or gameState.isLose() or plyCount == 0:
            return self.evaluationFunction(gameState)

        tNumGhosts = gameState.getNumAgents() - 1 ; cGhostIndex = tNumGhosts - numofGhosts + 1
        if numofGhosts > 1:
            for action in gameState.getLegalActions(cGhostIndex):
                eva.append(self.minFun(gameState.generateSuccessor(cGhostIndex, action), numofGhosts - 1, plyCount))
        else:
            for action in gameState.getLegalActions(cGhostIndex):
                eva.append(self.maxFun(gameState.generateSuccessor(cGhostIndex, action), tNumGhosts, plyCount - 1))

        return min(eva)

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** CS5368 YOUR CODE HERE ***"
        "PS. It is okay to define your own new functions. For example, value, minFun,maxFun"
        actions = []
        eva = []

        # import pdb; pdb.set_trace()
        for action in gameState.getLegalActions():
            actions.append(action)
            numofGhosts = gameState.getNumAgents() - 1
            eva.append(self.minFun(gameState.generateSuccessor(self.index, action), numofGhosts, self.depth))

        print("\n")
        print(gameState)
        return actions[eva.index(max(eva))]
        #need to return an action not a value
        # return self.maxFun(gameState, self.depth, gameState.getNumAgents() - 1)
        #use recursive helper function to make the best choice
        #every time everyone has taken an action, it's depth 1
        #return one of the legal actions
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def maxFun(self, gameState, numofGhosts, plyCount, alpha, beta):
        if (gameState.isWin() or gameState.isLose() or plyCount == 0):
            return self.evaluationFunction(gameState)

        x = - float('inf')
        for action in gameState.getLegalActions():
            x = max(x, self.minFun(gameState.generateSuccessor(self.index, action), numofGhosts, plyCount, alpha, beta))
            if x > beta:
                return x
            alpha = max(alpha, x)
        return x

    def minFun(self, gameState, numofGhosts, plyCount, alpha, beta):
        if (gameState.isWin() or gameState.isLose() or plyCount == 0):
            return self.evaluationFunction(gameState)

        tNumGhosts = gameState.getNumAgents() - 1
        cGhostIndex = tNumGhosts - numofGhosts + 1
        x = float('inf')
        if numofGhosts > 1:
            for action in gameState.getLegalActions(cGhostIndex):
                successorState = gameState.generateSuccessor(cGhostIndex, action)
                x = min(x, self.minFun(successorState, numofGhosts - 1, plyCount, alpha, beta))
                if x < alpha:
                    return x
                beta = min(beta, x)
        else:
            for action in gameState.getLegalActions(cGhostIndex):
                successorState = gameState.generateSuccessor(cGhostIndex, action)
                x = min(x, self.maxFun(successorState, tNumGhosts, plyCount - 1, alpha, beta))
                if x < alpha:
                    return x
                beta = min(beta, x)
        return x

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** CS5368 YOUR CODE HERE ***"
        "PS. It is okay to define your own new functions. For example, value, minFun,maxFun"
        actions = []
        eva = []

        alpha = - float('inf')
        beta = float('inf')
        x = - float('inf')
        for action in gameState.getLegalActions():
            actions.append(action)
            numofGhosts = gameState.getNumAgents() - 1
            successorState = gameState.generateSuccessor(self.index, action)
            x = max(x, self.minFun(successorState, numofGhosts, self.depth, alpha, beta))
            if x > beta:
                return x
            alpha = max(alpha, x)

            eva.append(x)

        return actions[eva.index(max(eva))]
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def maxFun(self, gameState, numofGhosts, plyCount):
        if gameState.isWin() or gameState.isLose() or plyCount == 0:
            return self.evaluationFunction(gameState)

        eva = []

        for action in gameState.getLegalActions():
            eva.append(self.minFun(gameState.generateSuccessor(self.index, action), numofGhosts, plyCount))

        return max(eva)

    def minFun(self, gameState, numofGhosts, plyCount):
        if gameState.isWin() or gameState.isLose() or plyCount == 0:
            return self.evaluationFunction(gameState)

        tNumGhosts = gameState.getNumAgents() - 1
        cGhostIndex = tNumGhosts - numofGhosts + 1
        sum = 0.0
        if numofGhosts > 1:
            for action in gameState.getLegalActions(cGhostIndex):
                sum += float(self.minFun(gameState.generateSuccessor(cGhostIndex, action), numofGhosts - 1, plyCount))
        else:
            for action in gameState.getLegalActions(cGhostIndex):
                sum += float(self.maxFun(gameState.generateSuccessor(cGhostIndex, action), tNumGhosts, plyCount - 1))
        # print("min eval:")
        # print(eva)
        return sum / (len(gameState.getLegalActions(cGhostIndex)))


    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** CS5368 YOUR CODE HERE ***"
        "PS. It is okay to define your own new functions. For example, value, minFun,maxFun"
        actions = []
        eva = []

        for action in gameState.getLegalActions():
            actions.append(action)
            numofGhosts = gameState.getNumAgents() - 1
            eva.append(self.minFun(gameState.generateSuccessor(self.index, action), numofGhosts, self.depth))

        return actions[eva.index(max(eva))]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    """
    Assigning new position as the current game state of pacman position and newfood as current game state and new ghost as current game state of ghosts states and new scared times as
    scared timer for ghosts state.
    Next finding closet food for the pacman to eat and the information of the ghost position.
    """
    "*** CS5368 YOUR CODE HERE ***"
    newPosition = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    #get closest food for the pacman to eat
    closestFood = None
    closestFoodDistance = float('inf')
    for f in newFood.asList():
        fDist = manhattanDistance(f, newPosition)
        if fDist < closestFoodDistance:
            closestFood = f
            closestFoodDistance = fDist

    t = 0
    if closestFood:
        minimumDistance = manhattanDistance(newPosition, closestFood)
        t -= minimumDistance * .25

    # information about the ghost positions
    ghostPos = []
    for ghostState in newGhostStates:
        g = ghostState.configuration.pos
        ghostPos.append(g)

    scaredGhostIndex = newScaredTimes.index(max(newScaredTimes))
    ghostScared = newScaredTimes[scaredGhostIndex]
    cGhostDist = float('inf')
    for g in ghostPos:
        dist = manhattanDistance(newPosition, g)
        if dist < cGhostDist:
            cGhostDist = dist
    if not ghostScared and cGhostDist <= 3:
        t -= (3 - cGhostDist) * 1000
    else:
        for time in newScaredTimes:
            scaredGhostPos= newGhostStates[newScaredTimes.index(time)].configuration.pos
            distToScaredGhost = manhattanDistance(newPosition, scaredGhostPos)
            if time > 0 and distToScaredGhost < 10:
                t += distToScaredGhost

    t += currentGameState.data.score

    if newPosition == currentGameState.getPacmanPosition():
        t -= 1

    return t
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
