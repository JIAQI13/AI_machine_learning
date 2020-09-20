from rl_glue import BaseAgent
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tiles3 import *


class TileCodingAgent(BaseAgent):

    def __init__(self):
        self.alpha = 0.01/50
        self.gamma = 1.0
        self.current_state = 0
        self.last_state = 0
        self.weights = None

        self.tilings = 50
        self.iht = IHT(1001)
        self.features = {}

    def agent_init(self):
        self.current_state = 0
        self.last_state = 0
        self.weights = np.zeros(1001)


    def agent_start(self, state):
        self.current_state = 500
        self.last_state = self.current_state
        choose = np.random.randint(1, 100)
        if choose < 51:
            action = np.random.randint(1, 100)
        else:
            action = np.random.randint(-100, -1)
        return action

    def agent_step(self, reward, state):
        self.current_state = state
        choose = np.random.randint(1, 100)
        if choose < 51:
            action = np.random.randint(1, 100)
        else:
            action = np.random.randint(-100, -1)
        feature = self.get_feature_vector(self.current_state)
        tderror = self.alpha + (reward + self.gamma * self.get_value(self.current_state, self.weights) - self.get_value(
            self.last_state, self.weights))
        self.weights = np.add(self.weights, tderror * self.get_feature_last_state(self.last_state))
        # self.weights += (TD_error * get_feature_vector(last_state))
        self.last_state = self.current_state
        return action

    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        tderror = self.alpha * (reward - self.get_value(self.last_state, self.weights))
        weights = np.add(self.weights, tderror * self.get_feature_vector(self.last_state))
        # weights += (TD_error * get_feature_vector(last_state))
        return

    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        if in_message == 'RMSE':
            x = np.zeros(1001)
            for state in range(1001):
                x[state] = self.get_value(state, self.weights)
            return x
        else:
            return "I dont know how to respond to this message!!"

    def get_value(self, state, weights):
        features = self.get_feature_vector(state)
        return np.dot(weights, features)

    def get_feature_vector(self, state):
        if state in self.features:
            return self.features[state]
        else:
            temp = np.zeros(1001)
            mytiles = tiles(self.iht, self.tilings, [float(state) / 200])
            for tile in mytiles:
                temp[tile] = 1
            self.features[state] = temp
            return self.features[state]


