
import matplotlib.pyplot as plt
import numpy as np
import pickle
import rndmwalk_policy_evaluation
from gambler_env import Environment
from Tilecoding import TileCodingAgent
from mc_agent import SemigradientTDAgent
from rndmwalk_policy_evaluation import *

from rl_glue import RLGlue

if __name__ == "__main__":
    num_episodes = 500
    max_steps = 1000
    num_runs = 30
    seed = 5678

    # Create and pass agent and environment objects to RLGlue
    environment = Environment()
    agents = [SemigradientTDAgent, TileCodingAgent]

    data = np.zeros((len(agents), num_runs, num_episodes))

    #true_value = np.load("TrueValueFunction.npy")
    true_value = rndmwalk_policy_evaluation.compute_value_function()
    #np.save("TrueValueFunction", true_value)

    for agent_number, agent in enumerate(agents):
        rlglue = RLGlue(environment, agent)
        for run in range(num_runs):
            np.random.seed(run)
            rlglue.rl_init()
            run_RMSE_array = np.zeros(num_episodes)
            for episode in range(num_episodes):
                if episode % (num_episodes / 10) == 0:
                    percent = float(episode) / num_episodes * 100
                    print(str(episode) + "/" + str(num_episodes) + "   " + str(percent) + "% of run")
                    print(run_RMSE_array[episode - 1])

                rlglue.rl_episode(max_steps)

                episode_values = rlglue.rl_agent_message("RMSE")
                RMSE = np.sqrt(np.mean((true_value[1:] - episode_values[1:]) ** 2))
                run_RMSE_array[episode] = RMSE

            data[agent_number - 1][run] = run_RMSE_array
            # print(np.random.randint(100,1000)) # Check that randomness same across episodes

        plt.plot(np.mean(data[agent_number - 1], axis=0), label=agent)

    plt.xlabel("Episodes")
    plt.ylabel("RMSE")
    plt.legend()
    #plt.savefig("random_walk.png")
    plt.show()



