import numpy as np
from tqdm import trange
from Q3 import ADP, print_probs
import pickle

import numpy as np
import pprint

class State():
    """
    Represents a state or a point in the grid.
    coord: coordinate in grid world
    """
    def __init__(self, coord):
        self.coord = coord
        self.is_terminal = self.isTerminal()
        self.reward = self._getReward()

    def __str__(self):
        return f"({self.coord[0]},{self.coord[1]})"

    # Returns if the current state is a terminal state
    def isTerminal(self):
        if (self.coord == (4,3)) or (self.coord == (4,2)):
            return True
        return False
    
    # Returns the reward of current state
    def _getReward(self):
        if self.coord == (4,3):
            return 1
        elif self.coord == (4,2):
            return -1
        return -0.04

    # Return the reward for stepping into this state - R(s)
    def getReward(self):
        return self.reward


##############################################################################

class GLIEAgent():
    """
    Implementation of a Agent that uses GLIE for the Grid World Problem
    """
    def __init__(self, gamma, P, actions):
        self.gamma = gamma
        self.P = P
        # of states and actions for the grid world problem
        self.num_actions = len(actions)
        self.actions = actions

    # Convert list index to world coordinates
    def _indexToCoords(self, i, j):
        x = j + 1
        y = 3 - i
        return (x, y)
    
    # Convert list index to world coordinates
    def cordination2Index(self, coord):
        x, y = coord
        j = x - 1
        i = 3 - y
        return (i,j)

    def initUPiAndN(self):
        pi = {}
        U = {}
        self.S = []
        for y in range(1, 4):
            for x in range(1, 5):
                # (2,2) is not a valid state
                if (x,y) == (2,2):
                    continue
                s = State((x,y))
                self.S.append(s)
                pi[s] = [0.25] * self.num_actions
                U[s] = 0
        # Initialize visit counts
        N = {(s, a): 0 for s in self.S for a in range(self.num_actions)}
        return U, N, pi

    def _printTable(self, U):
        # Fill the table with the estimated utilities
        table = np.zeros((3, 4))
        for s, u in U.items():
            table[self.cordination2Index(s.coord)] = u
        print("Value Function--------------------------")
        print(table)
    
    def _getTransitionProb(self, s, action, s_prime):
        return P.get((s.coord, self.actions[action]), {}).get(s_prime.coord, 0)

        # Gets the action values for a state by getting the expected return of taking each action
    def getUtilityForState(self, s, U):
        expected_utilities = []
        for action in range(self.num_actions):
            expected_utility = 0
            for s_prime in self.S:
                p = self._getTransitionProb(s, action, s_prime) # P(s'|s,a)
                expected_utility += p * U[s_prime]
            expected_utilities.append(expected_utility)
        return expected_utilities # gamma * (sum over all s': P(s'|s,a)) * V_s')) for each action
    
    def _f(self, U, N, state):
        f_values = []
        for action in range(self.num_actions):
            n = N[(state,action)]
            u = U[action]
            if n <= 5:
                value = 2
            else:
                value = u
            f_values.append(value)
        return f_values
        
    # Prints the policy as a grid of arrows
    def _printPolicy(self, pi):
        grid = np.empty([3,4], dtype=object)

        for state, actions in pi.items():
            i, j = self.cordination2Index(state.coord)
            actions = np.argmax(actions)
            grid[i, j] = actions

        # Convert actions to arrows
        arrow_grid = []
        for row_index, row in enumerate(grid):
            arrow_grid_row = []
            for col_index, action in enumerate(row):
                arrow_char = ''
                if ((row_index,col_index) == self.cordination2Index((4,3))) or ((row_index,col_index) == self.cordination2Index((4,2))) or ((row_index,col_index) == self.cordination2Index((2,2))):
                    arrow_grid_row.append(arrow_char)
                else:
                    if action == 0:
                        arrow_char += '↑'
                    elif action == 1:
                        arrow_char += '→'
                    elif action == 2:
                        arrow_char += '↓'
                    elif action == 3:
                        arrow_char += '←'
                    arrow_grid_row.append(arrow_char)
            arrow_grid.append(arrow_grid_row)

        print("Policy--------------------------")
        pprint.pprint(arrow_grid, width=50)
        print('\n')

    def GLIE(self, max_iterations):
        U, N, pi = self.initUPiAndN()
        # Iterate until convergence or maximum number of iterations
        for _ in trange(max_iterations):
            # Update the utilities using the GLIE scheme
            for state in self.S:
                u = U[state]
                expected_utilities = self.getUtilityForState(state, U)
                f_values = self._f(expected_utilities, N, state)
                new_best_action = np.argmax(f_values)
                max_f_value = f_values[new_best_action]
                U[state] = state.getReward() + self.gamma*max_f_value
                
                # Update the utilities and visit counts
                for s in self.S:
                    for a in range(self.num_actions):
                        N[(s, a)] += 1

                for action in range(self.num_actions):
                    if action != new_best_action:
                        pi[state][action] = 0
                    else:
                        pi[state][action] = 1

                # if state.isTerminal():
                #     continue
                # if (abs(u-U[state]) < 1e-10):
                #     self._printTable(U)
                #     self._printPolicy(pi)
                #     return U
        self._printTable(U)
        self._printPolicy(pi)
        return U

if __name__ == '__main__':

    # Define the discount factor
    gamma = 0.9

    actions = ("Move Up", "Move Right", "Move Down", "Move Left")

    # Probabilities learned from Q3
    try:
        with open('P.pkl', 'rb') as f:
            P = pickle.load(f)
    except:
        P = ADP(10000000, actions=actions)
        with open('P.pkl', 'wb') as f:
            pickle.dump(P, f)

    # printProbs(P)
    
    agent = GLIEAgent(gamma, P, actions)
    U = agent.GLIE(100000)

    