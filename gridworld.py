""" Gridworld
    with 2 magic squares that cause the agent to teleport accross the board
    purpose of this is to make the agent learn about the shortcut.
    
    Agent recieves reward of -1 at each step except for the terminal step or
    recieves a reward of 0, therefore agent will attempt to maximize it's 
    reward by minimizing the number of steps it takes to get off the grid world.
     
    State Space -> set of all states excluding the terminal state
    State Space Plus -> set of all states including the terminal state
""" 

import numpy as np
import matplotlib.pyplot as plt

class GridWorld(object):
    def __init__(self, m, n, magicSquares):
        self.grid = np.zeros((m,n))
        self.m = m
        self.n = n
        self.stateSpace = [i for i in range(self.m*self.n)]
        # removing the very last state because that is the terminal state we are excluding.
        self.stateSpace.remove(80)
        self.stateSpacePlus = [i for i in range(self.m*self.n)]
        self.actionSpace = {'U': -self.m, 'D': self.m,
                            'L': -1, 'R': 1}
        self.possibleActions = ['U', 'D', 'L', 'R']
        # do not need the function og MS, but need MSquares
        # do not need agent positions, but do need the State-trans Prob stores as dict.
        self.P = {}
        self.MagicSquares = magicSquares
        self.initP() # function to initialize those values.

    def initP(self):
        for state in self.stateSpace:
            for action in self.possibleActions:
                reward = -1
                state_ = state + self.actionSpace[action]
                if state_ in self.MagicSquares.keys():
                    state_ = self.MagicSquares[state_]
                if self.OffGridMove(state_, state): #remains in the same state (no action performed)
                    state_ = state
                #consider entering terminal state
                if self.isTerminalState(state_):
                    reward = 0 
                self.P[(state_, reward, state, action)] = 1



    # need to know that we are in the terminal state.
    def isTerminalState(self, state):
        return state in self.stateSpacePlus and state not in self.stateSpace
    
    
    def OffGridMove(self, newState, oldState):
        # if we move into a row not in the grid
        if newState not in self.stateSpacePlus:
            return True
        # if we're trying to wrap around to next row
        elif oldState % self.m == 0 and newState % self.m == self.m - 1:
            return True
        elif oldState % self.m == self.m - 1 and newState % self.m == 0:
            return True
        else:
            return False
 
# Printing value
def printV(V, grid):
    for idx, row in enumerate(grid.grid):
        for idy, _ in enumerate(row):
            state = grid.m * idx + idy
            print('%.2f' % V[state], end='\t')
        print('\n')
    print('-----------------------------')
    
def printPolicy(policy, grid):
    for idx, row in enumerate(grid.grid):
        for idy, _ in enumerate(row):
            state = grid.m * idx + idy
            if not grid.isTerminalState(state): # there is no action for the Terminal-state.
                if state not in grid.MagicSquares.keys(): # print out something different for magic squares, print the policy makes more sense.
                    print('%s' % policy[state], end='\t')
                else:
                    print('%s' % '--', end='\t')
            else:
                    print('%s' % '--', end='\t')
        print('\n')
    print('-----------------------------')

def evaluatePolicy(grid, V, policy, GAMMA, THETA):
    converged = False
    i= 0
    while not converged:
        DELTA = 0 # difference b/w the old and new value-function 
        for state in grid.stateSpace:
            i += 1
            oldV = V[state] # track of the OLD VALUE
            total = 0 # for how we add-up the value for the new state.
            weight = 1 / len(policy[state]) # probability from the policy
            for action in policy[state]:
                # iterate over state-transition probabilities
                for key in grid.P:
                    (newState, reward, oldState, act) = key
                    # find the entry in the stae-trans prob. that corresponds to the current state and action in the loop.
                    # We're given state and action, want new state and reward
                    if oldState == state and act == action:
                        # probability from our policy X probab. from the state-trans probs. X reward + GAMMA*(V(newSTate))
                        total += weight*grid.P[key]*(reward + GAMMA*V[newState])
                # update our value function
            V[state] = total 
            DELTA = max(DELTA, np.abs(oldV-V[state]))
            #check for convergence
            converged = True if DELTA < THETA else False
    print(i, "sweeps of state-space for evaluatePolicy")
    return V

def improvePolicy(grid, V, policy, GAMMA):
    stable = True
    newPolicy ={} # dictionary for out new policy
    i = 0
    for state in grid.stateSpace: # sweep over the state-space
        i += 1
        oldActions = policy[state] #saving our old policy
        value = []      # List           # as we want to take an argMax, we have to know the value of all actions in order to take the MAX.
        newAction = []  # List for the new actions
        for action in policy[state]: # iterate over our action-space
            weight = 1 / len(policy[state]) # Default random strategy.
            for key in grid.P: # iterate over our State-Transition probabilities
                (newState, reward, oldState, act) = key # un-packing
                if oldState==state and act==action:
                    # rather than increment of the value, 
                    # we do append to have a list of values for all the actions coz we need to find the max
                    value.append(np.round(weight*grid.P[key]*(reward+GAMMA*V[newState]), 2))
                    newAction.append(action) 
        # find our argMAX
        value = np.array(value) # converting value list to an array
        # means that you want the first value in the array, which is a list, and which contains the list of indices of condition-meeting cells.
        best = np.where(value == value.max())[0]
        bestActions = [newAction[item] for item in best] 
        newPolicy[state] = bestActions

        if oldActions != bestActions:
            stable = False
    print(i , "sweeps f state space in policy improvement")
    return stable, newPolicy
def iterateValues(grid, V, policy, GAMMA, THETA):
    converged = False
    i = 0 # dummy variable to count the no. of sweeps over state-space
    while not converged:
        DELTA = 0
        for state in grid.stateSpace:
            i += 1
            oldV = V[state] # keeping track of our old values
            newV = [] # list for the new values
            for action in grid.actionSpace:
                for key in grid.P:
                    (newState, reward, oldState, act) = key
                    if state == oldState and action == act:
                        newV.append(grid.P[key]*(reward+GAMMA*V[newState]))
            newV = np.array(newV)
            bestV = np.where(newV == newV.max())[0]
            bestState = np.random.choice(bestV)
            V[state] = newV[bestState]
            DELTA = max(DELTA, np.abs(oldV-V[state]))
            converged = True if DELTA < THETA else False
    # policy improvement (very similar)
    for state in grid.stateSpace:
        newValues = []
        actions = []
        i += 1
        for action in grid.actionSpace:
            for key in grid.P:
                (newState, reward, oldState, act) = key
                if state == oldState and action == act:
                    newValues.append(grid.P[key]*(reward+GAMMA*V[newState]))
            actions.append(action)
        newValues = np.array(newValues)
        bestActionIDX = np.where(newValues == newValues.max())[0]
        bestActions = actions[bestActionIDX[0]]
        policy[state] = bestActions
    print(i, 'sweeps of state space for value iteration')
    return V, policy

# there is no excel sheet to be accumulated 
if __name__ == '__main__':
    magicSquares = {18: 54, 63: 14} # Teleports
    env = GridWorld(9,9,magicSquares)

    # Model HyperParameters : these controls how fast the agent learns and 
    # how much it choses to value the potential future rewards.
    
    """
    Now we need some Convergence Criteria, so the 
    agent's estimate of the value function approaches the 
    true value function asymptotically that means you take 
    their difference, you're gonna get smaller and smaller number over time.
    """
    
    GAMMA = 1.0 # Discount factor: tells that our agent is going to be totally farsighted counting all future rewards equally
    THETA = 1e-6

    # Initial estimate of the value function
    V = {}
    # using ssPlus coz we wanna take into the account, the value of the terminal state.
    for state in env.stateSpacePlus:
        V[state] = 0
    
    policy ={} # can be anything you want 
    for state in env.stateSpace:
        policy[state] = env.possibleActions
    """
    stable = False
    i = 0
    while not stable:
        V = evaluatePolicy(env, V, policy, GAMMA, THETA)
        # printV(V, env)
        stable, policy = improvePolicy(env, V, policy , GAMMA)
    printV(V, env)
    print('\n--------------------------------\n')
    printPolicy(policy, env)
    print(i)
    """
    # 2 rounds of value iteration
    for i in range(2):
        V, policy = iterateValues(env, V, policy, GAMMA, THETA)

    printV(V, env)
    printPolicy(policy, env)