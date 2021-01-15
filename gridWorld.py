from math import *
from numpy import *
from random import *
import numpy as np
import matplotlib.pyplot as plt


# Tunable parameters:

gamma = 1
learning_rate = 0.85

class State(object):
    def __init__(self, i, j, is_wall=False, is_red=False, is_green=False):
        self.i = i
        self.j = j
        self.is_wall = is_wall
        self.is_red = is_red
        self.is_green = is_green
        self.neighbours = []
        self.state_value = 0
        #             north, east, south, west
        self.q_values = np.array([0.0, 0.0, 0.0, 0.0])

    def __str__(self):
        return '({}, {})'.format(self.i, self.j)

    def is_invalid_starting_space(self):
        if self.is_wall or self.is_red or self.is_green:
            return True

    def is_terminal(self):
        return self.is_red or self.is_green

    def get_max_q_index(self):
        best_q_values = np.argwhere(self.q_values == np.max(self.q_values))
        if len(best_q_values) > 1:
            return best_q_values[randint(0, len(best_q_values) - 1)][0]
        else:
            _max_q = np.argmax(self.q_values)
            return _max_q

    def get_max_q_value(self):
        return np.max(self.q_values)
    
    def get_neighbours(self, states, walls):
        if self.i >= 1:
            self.neighbours.append(states[self.i - 1][self.j])
        if self.i <= 9 - 2:
            self.neighbours.append(states[self.i + 1][self.j])
        if self.j >= 1:
            self.neighbours.append(states[self.i][self.j - 1])
        if self.j <= 9 - 2:
            self.neighbours.append(states[self.i][self.j + 1])
        for neighbour in self.neighbours:
            if str(neighbour) in walls:
                self.neighbours.remove(neighbour)
        add_current_state = 4 - len(self.neighbours)
        for i in range(add_current_state):
            self.neighbours.append(states[self.i][self.j])
        
    def get_reward_state(self, neighbour):
        if neighbour.is_green:
            reward = 50
        elif neighbour.is_red:
            reward = -50
        else:
            reward = (-1 + neighbour.state_value)
        return reward

def initialize_states():
    # This is the set of states, all initialised with default values
    states = [[State(j, i) for i in range(9)] for j in range(9)]

    # Now I make the walls black
    walls = []
    for j in range(2,7):
      states[1][j].is_wall = True
      walls.append(str(states[1][j]))

    for j in range(1, 5):
      states[7][j].is_wall = True
      walls.append(str(states[7][j]))

    for i in range(2, 6):
      states[i][6].is_wall = True
      walls.append(str(states[i][6]))

    states[6][5].is_red = True
    states[8][8].is_green = True
    
    # Calculate value function

    for row in states:
        for state in row:
            state.get_neighbours(states, walls)
    
    for _ in range(500):
        for row in states:
            for state in row:
                updated_value = 0
                if state.is_wall:
                    state.state_value = 0
                elif state.is_red:
                    state.state_value = -50
                elif state.is_green:
                    state.state_value = 50
                else:
                    for neighbour in state.neighbours:
                        reward = state.get_reward_state(neighbour)
                        updated_value += 1/4 * reward
                    state.state_value = updated_value
                     
    return states

# The reward function defines what reward I get for transitioning between the first and second state
def reward(s_1, s_2): 
  if (s_2.is_red):
    return -50
  elif (s_2.is_green):
    return 50
  else:
    return -1

""" the transition function takes state and action and results in a new state, depending on their attributes. The method takes the whole state-space as an argument (since the transition depends on the attributes of the states in the state-space), which could for example be the "states" matrix from above, the current state s from the state-space (with its attributes), and the current action, which takes the form of a "difference vector. For example, dx = 0, dy = 1 means: Move to the south. dx = -1, dy = 0 means: Move to the left"""
def transition(stsp, s, di, dj):
  if (s.is_red or s.is_green):
    return s
  elif (s.j + dj not in range(9) or s.i + di not in range(9)):
    return s
  elif (stsp[s.i + di][s.j + dj].is_wall):
    return s
  else:
    return stsp[s.i + di][s.j + dj]

def action_to_diff_vector(action):
    if action == 0:  # NORTH
        return -1, 0
    elif action == 1:  # EAST
        return 0, 1
    elif action == 2:  # SOUTH
        return 1, 0
    elif action == 3:  # WEST
        return 0, -1

def action_to_verbose(action):
    if action == 0:
        return 'NORTH'
    elif action == 1:
        return 'EAST'
    elif action == 2:
        return 'SOUTH'
    elif action == 3:
        return 'WEST'

def sarsa(state, next_state, action, next_state_action):
    return state.q_values[action] +\
            learning_rate * (reward(state, next_state) + gamma * next_state.q_values[next_state_action] - state.q_values[action])


def q_learning(state, next_state, action, next_state_action):
    next_state_q_value = next_state.get_max_q_value()
    return state.q_values[action] +\
            learning_rate * (reward(state, next_state) + gamma * next_state_q_value - state.q_values[action])
            

def run_code(use_q_learning = False, show_state_values = False):
    states = initialize_states()
    epsilon = 0.5
    decay = 0.999
    min_epsilon = 0.000000001

    if show_state_values:
        plot_state_values(states)

    for i in range(10000):
        # select a random starting state
        current_state = states[randint(0, 8)][randint(0, 8)]
        while current_state.is_invalid_starting_space():
            current_state = states[randint(0, 8)][randint(0, 8)]

        # iterate until reaching a terminal state
        epsilon = max(min_epsilon, epsilon * decay)
        while not current_state.is_terminal():
            if random() < epsilon:
                next_action = randint(0, 3)
            else:
                next_action = current_state.get_max_q_index()

            di, dj = action_to_diff_vector(next_action)
            next_state = transition(states, current_state, di, dj)

            if random() < epsilon:
                next_state_action = randint(0, 3)
            else:
                next_state_action = next_state.get_max_q_index()
                
            if use_q_learning:
                current_state.q_values[next_action] = q_learning(current_state, next_state, next_action, next_state_action)
            else:
                current_state.q_values[next_action] = sarsa(current_state, next_state, next_action, next_state_action)

            # print(current_state, next_state, action_to_verbose(next_action), di, dj)
            current_state = next_state

        if (i % 100 == 0):
            print(i)
        

    return states

def plot_state_values(states):
    final_grid = np.array([[states[j][i].state_value for i in range(9)] for j in range(9)])
    fig, ax = plt.subplots()
    im = ax.imshow(final_grid, cmap='coolwarm')
    ax.set_xticks(np.arange(9))
    ax.set_yticks(np.arange(9))
    ax.set_xticklabels([i for i in range(9)])
    ax.set_yticklabels([i for i in range(9)])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(9):
        for j in range(9):
            text = ax.text(j, i, '{:.2f}'.format(states[i][j].state_value),
                           ha="center", va="center", color="w")


    plt.title("State values for each cell")
    for i in range(9):
        str_ = ""
        for j in range(9):
            str_ += str(int(final_grid[i][j])) + ", "
    plt.show()

def plot_best_q_values_states(states):
    final_grid = np.array([[max(states[j][i].q_values) for i in range(9)] for j in range(9)])
    fig, ax = plt.subplots()
    im = ax.imshow(final_grid, cmap='coolwarm')
    ax.set_xticks(np.arange(9))
    ax.set_yticks(np.arange(9))
    ax.set_xticklabels([i for i in range(9)])
    ax.set_yticklabels([i for i in range(9)])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(9):
        for j in range(9):
            text = ax.text(j, i, '{:.2f}'.format(max(states[i][j].q_values)),
                           ha="center", va="center", color="w")


    plt.title("Best q values for each cell")
    for i in range(9):
        str_ = ""
        for j in range(9):
            str_ += str(int(final_grid[i][j])) + ", "
    plt.show()

def q_to_vector(states):
    U = []
    V = []
    for i in range(len(states)):
        u = []
        v = []
        for j in range(len(states[0])):
            if states[i][j].is_wall or states[i][j].is_red or states[i][j].is_green:
                u.append(0)
                v.append(0)
            else:
                index = states[i][j].get_max_q_index()
                if index == 0:
                    u.append(0)
                    v.append(0.1)
                elif index == 1:
                    u.append(0.1)
                    v.append(0)
                elif index == 2:
                    u.append(0)
                    v.append(-0.1)
                else:
                    u.append(-0.1)
                    v.append(0)
        U.append(u)
        V.append(v)
    return U, V

def display_optimal_policy_quiver(states):
    final_grid = np.array([[max(states[j][i].q_values) for i in range(9)] for j in range(9)])
    X = np.arange(0, 9, 1)
    Y = np.arange(0, 9, 1)
  
    U, V = q_to_vector(states)

    fig, ax = plt.subplots()
 
    q = ax.quiver(X, Y, U, V)
    ax.invert_yaxis()

    plt.show()


if __name__ == '__main__':
    end_states = run_code(use_q_learning=False, show_state_values = False)
    
    plot_best_q_values_states(end_states)
    display_optimal_policy_quiver(end_states)
    plt.gca().invert_yaxis()


