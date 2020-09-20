#!/usr/bin/env python

import numpy as np
from agent_hw6 import Agent

from rl_glue import RLGlue
from env_hw6 import Environment


def question_1():
    # Specify hyper-parameters

    agent = Agent()
    environment = Environment()
    rlglue = RLGlue(environment, agent)

    num_episodes = 200
    num_runs = 50
    max_eps_steps = 100000

    steps = np.zeros([num_runs, num_episodes])

    for r in range(num_runs):
        print("run number : ", r)
        rlglue.rl_init()
        for e in range(num_episodes):
            rlglue.rl_episode(max_eps_steps)
            steps[r, e] = rlglue.num_ep_steps()
            # print(steps[r, e])
    np.save('steps', steps)

if __name__ == "__main__":
    question_1()
    print("Done")
