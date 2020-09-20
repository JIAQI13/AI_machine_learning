
import numpy.random as rnd
from rl_glue import BaseAgent
import numpy as np
from tile3 import *

class SemigradientTDAgent(BaseAgent):
    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.last_tile = None
        self.weights = None
        self.values = None
        self.Z = None

        self.tilings = 8
        self.alpha = 0.1 / self.tilings
        self.gamma = 1.0
        self.lambdah = 0.9
        self.epsilon = 0.0
        self.shape = [8, 8]
        self.iht = IHT(4096)

    def agent_init(self):

        self.values = np.zeros([self.shape[0], self.shape[1], 3])  # size of tiling shape and depth of # actions
        self.weights = np.random.uniform(-0.001, 0.0, 4096)  # eClass said use 4096, same as IHT size

    def agent_start(self, state):
        position = self.shape[0] * (state[0] + 1.2) / (
                        1.2 + 0.5)  # add lower bound 1.2 to get positives only, divide by range
        velocity = self.shape[1] * (state[1] + 0.07) / (0.07 + 0.07)
        tile = [position, velocity]
        action = self.get_epsilon_action(tile)
        self.last_state = state
        self.last_action = action
        self.last_tile = tile
        self.Z = np.zeros(len(weights))
        return action

    def agent_step(self,reward, state):
        TD_error = reward
        for i in self.F(self.last_tile, self.last_action):
            TD_error -= self.weights[i]
            self.Z[i] = 1.0
        position = self.shape[0] * (state[0] + 1.2) / (
                        1.2 + 0.5)  # add lower bound 1.2 to get positives only, divide by range
        velocity = self.shape[1] * (state[1] + 0.07) / (0.07 + 0.07)
        tile = [position, velocity]

        action = self.get_epsilon_action(tile)

        activated = np.zeros(len(self.weights))

        for i in self.F(tile, action):
            activated[i] = 1.0
            TD_error += self.gamma * self.weights[i]

        self.weights += self.alpha * TD_error * self.Z
        self.Z = self.gamma * self.lambdah * self.Z

        self.last_state = state
        self.last_action = action
        self.last_tile = tile

        return action

    def agent_end(self,reward):
        TD_error = reward
        for i in self.F(self.last_tile, self.last_action):
            TD_error -= self.weights[i]
            self.Z[i] = 1.0
        self.weights += self.alpha * TD_error * self.Z
        return

    def agent_message(self, in_message):  # returns string, in_message: string
        if (in_message == 'RMSE'):
            estimated_values = np.zeros(1001)
            for state in range(1001):
                estimated_values[state] = get_value(state, self.weights)
            return estimated_values
        else:
            return "I don't know what to return!!"

    def get_epsilon_action(self, state):
        if np.random.uniform() < self.epsilon:  # explore
            action = np.random.randint(3)
            print("Yeaaaaaaaaaaaaaaaaaaa this probably shouldn't have happened when eps=0, plz fix")
        else:
            options = []
            for action in range(3):
                tiles123 = self.F(state, action)
                options.append(np.sum(self.weights[tiles123]))
            options = np.array(options)
            action = np.argmax(options)
        return action

    def F(self, state, action):
        return tiles(self.iht, self.tilings, state, [action])

    def rand_in_range(self, max):  # returns integer, max: integer
        return rnd.randint(max)

    def rand_un(self):  # returns floating point
        return rnd.uniform()

    def rand_norm(self, mu, sigma):  # returns floating point, mu: floating point, sigma: floating point
        return rnd.normal(mu, sigma)
