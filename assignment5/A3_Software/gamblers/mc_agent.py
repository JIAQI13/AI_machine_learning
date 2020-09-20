from rl_glue import BaseAgent
import numpy as np


class SemigradientTDAgent(BaseAgent):
    def __init__(self):
        self.alpha = 0.5
        self.gamma = 1.0
        self.current_state = 0
        self.last_state = 0
        self.weights = None

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
        feature = np.zeros(1001)
        feature[state] = 1
        feature_last_state = np.zeros(1001)
        feature_last_state[self.last_state] = 1
        tderror = self.alpha+(reward + self.gamma * self.get_value(self.current_state,self.weights)-self.get_value(
            self.last_state, self.weights))
        self.weights = np.add(self.weights, tderror * feature_last_state)
        self.last_state = self.current_state
        return action

    def agent_end(self, reward):
        feature_last_state = np.zeros(1001)
        feature_last_state[self.last_state] = 1
        tderror = self.alpha * (reward - self.get_value(self.last_state,self.weights))
        self.weights = np.add(self.weights, tderror * feature_last_state)
        return None

    def agent_message(self, in_message):
        if in_message == 'RMSE':
            x = np.zeros(1001)
            for i in range(1001):
                x[i]= self.get_value(i, self.weights)
            return x
        else:
            return "I dont know how to respond to this message!!"

    def get_value(self, state, weights):
        features = np.zeros(1001)
        features[state] = 1
        return np.dot(weights, features)
