import numpy as np
from tqdm import trange
from Q3 import ADP, print_probs
import pickle
import pprint

class State:
    def __init__(self, coord):
        self.coord = coord
        self.is_terminal = self.is_terminal_state()
        self.reward = self.get_reward()

    def __str__(self):
        return f"({self.coord[0]},{self.coord[1]})"

    def is_terminal_state(self):
        return self.coord in [(4, 3), (4, 2)]

    def get_reward(self):
        if self.coord == (4, 3):
            return 1
        elif self.coord == (4, 2):
            return -1
        return -0.04

class GLIEAgent:
    def __init__(self, gamma, P, actions):
        self.gamma = gamma
        self.P = P
        self.num_actions = len(actions)
        self.actions = actions

    def index_to_coords(self, i, j):
        x = j + 1
        y = 3 - i
        return (x, y)

    def coords_to_index(self, coord):
        x, y = coord
        j = x - 1
        i = 3 - y
        return (i, j)

    def initialize_values_and_policy(self):
        pi = {}
        U = {}
        self.S = []
        for y in range(1, 4):
            for x in range(1, 5):
                if (x, y) == (2, 2):
                    continue
                s = State((x, y))
                self.S.append(s)
                pi[s] = [0.25] * self.num_actions
                U[s] = 0
        N = {(s, a): 0 for s in self.S for a in range(self.num_actions)}
        return U, N, pi

    def print_value_table(self, U):
        table = np.zeros((3, 4))
        for s, u in U.items():
            table[self.coords_to_index(s.coord)] = u
        print("Value Function--------------------------")
        print(table)

    def get_transition_prob(self, s, action, s_prime):
        return self.P.get((s.coord, self.actions[action]), {}).get(s_prime.coord, 0)

    def get_expected_utilities(self, s, U):
        expected_utilities = []
        for action in range(self.num_actions):
            expected_utility = 0
            for s_prime in self.S:
                p = self.get_transition_prob(s, action, s_prime)
                expected_utility += p * U[s_prime]
            expected_utilities.append(expected_utility)
        return expected_utilities

    def compute_f_values(self, U, N, state):
        f_values = []
        for action in range(self.num_actions):
            n = N[(state, action)]
            u = U[action]
            value = 2 if n <= 5 else u
            f_values.append(value)
        return f_values

    def print_policy(self, pi):
        grid = np.empty([3, 4], dtype=object)

        for state, actions in pi.items():
            i, j = self.coords_to_index(state.coord)
            actions = np.argmax(actions)
            grid[i, j] = actions

        arrow_grid = []
        for row_index, row in enumerate(grid):
            arrow_grid_row = []
            for col_index, action in enumerate(row):
                arrow_char = ''
                if ((row_index, col_index) == self.coords_to_index((4, 3))) or ((row_index, col_index) == self.coords_to_index((4, 2))) or ((row_index, col_index) == self.coords_to_index((2, 2))):
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
        U, N, pi = self.initialize_values_and_policy()
        for _ in trange(max_iterations):
            for state in self.S:
                u = U[state]
                expected_utilities = self.get_expected_utilities(state, U)
                f_values = self.compute_f_values(expected_utilities, N, state)
                new_best_action = np.argmax(f_values)
                max_f_value = f_values[new_best_action]
                U[state] = state.get_reward() + self.gamma * max_f_value

                for s in self.S:
                    for a in range(self.num_actions):
                        N[(s, a)] += 1

                for action in range(self.num_actions):
                    pi[state][action] = 1 if action == new_best_action else 0

        self.print_value_table(U)
        self.print_policy(pi)
        return U

gamma = 0.9
actions = ("Move Up", "Move Right", "Move Down", "Move Left")
try:
    with open('P.pkl', 'rb') as f:
        P = pickle.load(f)
except:
    P = ADP(10000000, actions=actions)
    with open('P.pkl', 'wb') as f:
        pickle.dump(P, f)

agent = GLIEAgent(gamma, P, actions)
U = agent.GLIE(100000)
