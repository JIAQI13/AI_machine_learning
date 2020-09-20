import numpy as np
from maze_env import Environment
from dyna-q_agent import Agent
from rl_glue import RLGlue

if __name__ == "__main__":
    num_episodes = 50
    max_steps = 2500
    num_runs = 10
    seed = 5678

    n_sweep = [0, 5, 50]

    data = np.zeros((num_episodes, num_runs))
    rlglue = RLGlue(Environment, Agent)

    for n in n_sweep:
        rlglue.rl_agent_message(["n", n])
        rlglue.rl_agent_message(["alpha", 0.1])

        for run in range(num_runs):
            np.random.seed(366609 + run)
            # print("Seed = " + str(366 + run))
            print("run number: " + str(run))
            rlglue.rl_init()
            for episode in range(num_episodes):
                rlglue.rl_episode(max_steps)
                data[episode][run] = rlglue._num_steps()
                # if episode == 0:
                #     print("First Episode Length = " + str(RL_num_steps()))

        average_over_runs = np.mean(data, axis=1)  # axis=1 does mean by row, ie per episode
        print(average_over_runs)
        average_over_runs[0] = None  # do not plot first episode
        filename = "output" + str(n)
        np.save(filename, average_over_runs)