import numpy as np
import pprint

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class State:
    
    def __init__(self, coord):
        self.coordination = coord
        self.action_state_transitions = self.get_action_state_transitions()
        self.is_terminal = self.is_terminal_state()
        self.reward = self.get_reward()

    def __str__(self):
        return f"({self.coordination[0]},{self.coordination[1]})"

    def is_terminal_state(self):
        return self.coordination in [(4, 3), (4, 2)]

    def get_reward(self):
        if self.coordination == (4, 3):
            return 1
        elif self.coordination == (4, 2):
            return -1
        return -0.04

    def get_action_state_transitions(self):
        if self.is_terminal_state():
            return {UP: (), LEFT: (), RIGHT: (), DOWN: ()}
        
        action_state_transitions = {}

        x, y = self.coordination

        action_state_transitions[UP] = (
            self.validate_coordinates(x, y + 1),
            self.validate_coordinates(x + 1, y),
            self.validate_coordinates(x - 1, y)
        )
        action_state_transitions[RIGHT] = (
            self.validate_coordinates(x + 1, y),
            self.validate_coordinates(x, y + 1),
            self.validate_coordinates(x, y - 1)
        )
        action_state_transitions[DOWN] = (
            self.validate_coordinates(x, y - 1),
            self.validate_coordinates(x + 1, y),
            self.validate_coordinates(x - 1, y)
        )
        action_state_transitions[LEFT] = (
            self.validate_coordinates(x - 1, y),
            self.validate_coordinates(x, y + 1),
            self.validate_coordinates(x, y - 1)
        )

        return action_state_transitions

    def validate_coordinates(self, x, y):
        if (x, y) == (2, 2) or x < 1 or x > 4 or y < 1 or y > 3:
            return self.coordination
        return (x, y)

    def get_next_state_likelihood(self, a, s_prime):
        p = 0
        if s_prime.coordination in self.action_state_transitions[a]:
            if self.action_state_transitions[a][0] == s_prime.coordination:
                p += 0.8
            if self.action_state_transitions[a][1] == s_prime.coordination:
                p += 0.1
            if self.action_state_transitions[a][2] == s_prime.coordination:
                p += 0.1
        return p

class ValueIterationAgent:
    
    def __init__(self, gamma):
        self.gamma = gamma
        self.num_states = 12
        self.num_actions = 4

    def print_state_values(self, V):
        grid = np.zeros([3, 4])
        for state, value in V.items():
            i, j = self.coords_to_index(state.coordination)
            grid[i, j] = value
        print("Value Function--------------------------")
        pprint.pprint(grid)
        print('\n')

    def index_to_coords(self, i, j):
        x = j + 1
        y = 3 - i
        return (x, y)
    
    def coords_to_index(self, coord):
        x, y = coord
        j = x - 1
        i = 3 - y
        return (i, j)
    
    def print_policy(self, pi):
        grid = np.empty([3, 4], dtype=object)
        for state, actions in pi.items():
            i, j = self.coords_to_index(state.coordination)
            actions = np.argwhere(actions == np.max(actions)).flatten().tolist()
            grid[i, j] = actions

        arrow_grid = []
        for row_index, row in enumerate(grid):
            arrow_grid_row = []
            for col_index, actions in enumerate(row):
                arrow_char = ''
                if ((row_index, col_index) == self.coords_to_index((4, 3))) or ((row_index, col_index) == self.coords_to_index((4, 2))) or ((row_index, col_index) == self.coords_to_index((2, 2))):
                    arrow_grid_row.append(arrow_char)
                else:
                    for action in actions:
                        if action == UP:
                            arrow_char += '↑'
                        elif action == RIGHT:
                            arrow_char += '→'
                        elif action == DOWN:
                            arrow_char += '↓'
                        elif action == LEFT:
                            arrow_char += '←'
                    arrow_grid_row.append(arrow_char)
            arrow_grid.append(arrow_grid_row)

        print("Policy--------------------------")
        pprint.pprint(arrow_grid, width=50)
        print('\n')

    def init_S_V_and_pi(self):
        self.S = []
        V = {}
        pi = {}
        for y in range(1, 4):
            for x in range(1, 5):
                if (x, y) == (2, 2):
                    continue
                s = State((x, y))
                self.S.append(s)
                V[s] = 0
                pi[s] = [0.25] * self.num_actions
        return V, pi

    def get_action_values_for_state(self, s, V):
        action_values = []
        for action in range(self.num_actions):
            action_value = 0
            for s_prime in self.S:
                p = s.get_next_state_likelihood(action, s_prime)
                action_value += p * V[s_prime]
            action_values.append(self.gamma * action_value)
        return action_values

    def value_iterate(self):
        V, pi = self.init_S_V_and_pi()
        theta = 1e-6
        while True:
            delta = 0
            for s in self.S:
                v = V[s]
                action_values = np.round(self.get_action_values_for_state(s, V), 10)
                V[s] = s.get_reward() + max(action_values)
                new_best_actions = np.argwhere(action_values == np.max(action_values)).flatten().tolist()
                for action in range(self.num_actions):
                    if action not in new_best_actions:
                        pi[s][action] = 0
                    else:
                        pi[s][action] = 1
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                self.print_state_values(V)
                self.print_policy(pi)
                break

agent = ValueIterationAgent(0.9)
agent.value_iterate()
