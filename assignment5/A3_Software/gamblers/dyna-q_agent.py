from rl_glue import BaseAgent
import numpy as np
import numpy.random as rnd
from random import *


class Agent(BaseAgent):

    def __init__(self):
        self.alpha = None
        self.gamma = 0.95
        self.current_state = 0
        self.n = 0
        self.epsilon = 0.1

        self.Q = None
        self.model = None
        self.S = None
        self.last_action = None
        self.S_ = None
        self.previous_states = None

    def agent_init(self):
        self.Q = np.zeros((6, 9, 4))
        self.model = np.zeros((6, 9, 4), object)
        self.previous_states = {}

    def agent_start(self, state):
        self.S = state

        if self.rand_un() <= self.epsilon:
            action = self.rand_in_range(4)
        else:
            action = np.argmax(self.Q[self.S[0]][self.S[1]])

        self.last_action = int(action)

        return action

    def agent_step(self, reward, state):
        self.S_ = state
        self.S_ = state

        if (self.S[0], self.S[1]) not in self.previous_states:
            self.previous_states[(self.S[0], self.S[1])] = set()
        self.previous_states[(self.S[0], self.S[1])].add(self.last_action)

        self.Q[self.S[0]][self.S[1]][self.last_action] += self.alpha * (reward + self.gamma * max(self.Q[self.S_[0]][self.S_[1]]) - self.Q[self.S[0]][self.S[1]][self.last_action])
        self.model[self.S[0]][self.S[1]][self.last_action] = (reward, self.S_[0], self.S_[1])

        for i in range(self.n):
            S_planning = random.choice(self.previous_states.keys())
            A_planning = random.sample(self.previous_states[S_planning], 1)
            reward_planning, x_planning, y_planning = self.model[S_planning[0]][S_planning[1]][A_planning[0]]
            self.Q[S_planning[0]][S_planning[1]][A_planning[0]] += self.alpha * (reward_planning + self.gamma * max(self.Q[x_planning][y_planning]) - self.Q[S_planning[0]][S_planning[1]][A_planning[0]])

        if self.rand_un() < self.epsilon:
            action = self.rand_in_range(4)
        else:
            action = np.argmax(self.Q[self.S[0]][self.S[1]])

        self.S = self.S_
        self.last_action = action

        return action

    def agent_end(self, reward):
        self.Q[self.S[0]][self.S[1]][self.last_action] += self.alpha * (reward - self.Q[self.S[0]][self.S[1]][self.last_action])

        return None

    def agent_message(self, in_message):
        if in_message[0] == "n":
            self.n = in_message[1]
        elif in_message[0] == "alpha":
            self.alpha = in_message[1]
        else:
            return "I don't know what to return!!"

    def rand_in_range(self,max):  # returns integer, max: integer
        return rnd.randint(max)

    def rand_un(self):  # returns floating point
        return rnd.uniform()

    def rand_norm(self,mu, sigma):  # returns floating point, mu: floating point, sigma: floating point
        return rnd.normal(mu, sigma)
