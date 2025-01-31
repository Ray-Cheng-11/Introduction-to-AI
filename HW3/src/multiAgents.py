from util import manhattanDistance
from game import Directions
import random
import util
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
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min(
            [manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food)
                                  for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(
            newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(
            newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (Part 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)
        """
        Minimax Search
        1. Initialize an empty list called 'all_choice' to record the score of each step in this recursion.
        2. The recursion stop when the depth equals to the given depth, or the the game is either a winning state 
        or losing state, and return the score calculated by self.evaluationFunction.
        3. Run for loop to do recursion to evaluate the score of each legal action and store the results in the 'all_choice'.
        4. After getting all the evaluated scores, determine if the 'index' represents ghosts, return the minimum value of 'all_choice'; 
        otherwise, return the maximum value if the recursion not in the zero depth, and if it is in the zero depth, 
        then return the action which has the highest score.
        """
        def minimax(state, depth, index):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            all_choice = []
            legal_actions = state.getLegalActions(index)
            if Directions.STOP in legal_actions:
                # we don't want the pacman to stop in some cases.
                legal_actions.remove(Directions.STOP)
            for action in legal_actions:
                nextState = state.getNextState(index, action)
                if index != gameState.getNumAgents()-1:
                    all_choice.append(minimax(nextState, depth, index+1))
                else:
                    all_choice.append(minimax(nextState, depth+1, 0))
            if index:
                return min(all_choice)
            else:
                return max(all_choice) if depth else legal_actions[all_choice.index(max(all_choice))]

        return minimax(gameState, 0, 0)
        # End your code (Part 1)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)
        """
        Alpha-Beta Pruning
        1. Initialize 'initVal' to positive infinity if the index represents ghosts, or to negative infinity if the index represents pacman.
        2. The recursion stop when the depth equals to the given depth, or the the game is either a winning state or losing state, and return the score calculated by self.evaluationFunction.
        3. If the depth and index both aren't zero, run for loop to do recursion to evaluate the score of each legal action. If index is zero, 'initVal' equals to the maximum value between 'initVal' and the returned score, and update the 'alpha' if 'initVal' is larger than 'alpha'; else, 'initVal' equals to the minimum value between 'initVal' and the returned score, and update the 'beta' if 'initVal' is less than 'beta'.
        4. If 'alpha' is larger than 'beta', break the for loop and return the 'initVal'; if not, do the next for loop and return 'initVal' after finishing.
        5. If the depth and index are both zero, initialize an empty list 'all_choice' to record the evaluated score, and run for loop to get scores and store in the 'all_choice' and update the 'initVal' and 'alpha'.
        6. If 'alpha' is larger than 'beta', break the for loop; if not, do the next for loop. After finishing, return the action which value is equal to 'alpha'.
        """
        def AlphaBeta(state, depth, index, alpha=float('-inf'), beta=float('inf')):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            initVal = float('inf') if index else float('-inf')
            legal_actions = state.getLegalActions(index)
            if Directions.STOP in legal_actions:
                # we don't want the pacman to stop in some cases.
                legal_actions.remove(Directions.STOP)
            if depth or index:
                for action in legal_actions:
                    nextState = state.getNextState(index, action)
                    if index == 0:
                        initVal = max(initVal, AlphaBeta(
                            nextState, depth, 1, alpha, beta))
                        alpha = max(alpha, initVal)
                    else:
                        if index == gameState.getNumAgents() - 1:
                            initVal = min(initVal, AlphaBeta(
                                nextState, depth+1, 0, alpha, beta))
                        else:
                            initVal = min(initVal, AlphaBeta(
                                nextState, depth, index+1, alpha, beta))
                        beta = min(beta, initVal)
                    if alpha > beta:
                        break
                return initVal
            else:
                all_choice = []
                for action in legal_actions:
                    all_choice.append(AlphaBeta(state.getNextState(
                        0, action), depth, 1, alpha, beta))
                    initVal = max(initVal, all_choice[-1])
                    alpha = max(alpha, initVal)
                    if alpha > beta:
                        break
                return legal_actions[all_choice.index(alpha)]

        return AlphaBeta(gameState, 0, 0)
        # End your code (Part 2)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        """
        1. The steps are the same as Minimax Search, but return the mean value of 'all_choice' rather than minimal
        value if the index represents ghosts.
        """
        def expectimax(state, depth, index):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            all_choice = []
            legal_actions = state.getLegalActions(index)
            if Directions.STOP in legal_actions:
                # we don't want the pacman to stop in some cases.
                legal_actions.remove(Directions.STOP)
            for action in legal_actions:
                nextState = state.getNextState(index, action)
                if index != gameState.getNumAgents()-1:
                    all_choice.append(expectimax(nextState, depth, index+1))
                else:
                    all_choice.append(expectimax(nextState, depth+1, 0))
            if index:
                return sum(all_choice)/len(all_choice)
            else:
                if depth:
                    return max(all_choice)
                else:
                    return legal_actions[all_choice.index(max(all_choice))]

        return expectimax(gameState, 0, 0)
        # End your code (Part 3)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Part 4).
    """
    # Begin your code (Part 4)
    """
    Initialize variables and get the current game state we want.
    """
    score = currentGameState.getScore()
    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsuleList = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    minFoodDist = float('inf')
    minCapsuleDist = float('inf')
    scaredGhostDist = float('inf')

    """
    Calculate the minimal position of food, capsule, and scared ghosts.
    """
    for food in foodList:
        minFoodDist = min(minFoodDist, manhattanDistance(pos, food))
    for capsule in capsuleList:
        minCapsuleDist = min(minCapsuleDist, manhattanDistance(pos, capsule))
    for ghost in ghostStates:
        if ghost.scaredTimer > 0:
            scaredGhostDist = min(
                scaredGhostDist, manhattanDistance(pos, ghost.getPosition()))

    """
    My evaluation function consider the current score, minimal food distance, minimal capsule distance, and minimal scared ghost distance.
    """
    return score+(10/(minFoodDist))+(20/(minCapsuleDist))+(200/(scaredGhostDist))
    # End your code (Part 4)


# Abbreviation
better = betterEvaluationFunction
