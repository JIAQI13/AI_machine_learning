from rl_glue import BaseEnvironment
import copy
import numpy as np


class Environment(BaseEnvironment):
    """
    Slightly modified Gambler environment -- Example 4.3 from
    RL book (2nd edition)

    Note: inherit from BaseEnvironment to be sure that your Agent class implements
    the entire BaseEnvironment interface
    """

    def __init__(self):
        """Declare environment variables."""
        self.start = np.array([2, 0])
        self.goal = np.array([0, 8])
        self.walls = [(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)]
        self.current_state = None

    def env_init(self):
        self.current_state = self.start

    def env_start(self):
        self.current_state = self.start
        return self.current_state

    def env_step(self, action):
        new_state = copy.deepcopy(self.current_state)

        if action == 0:  # south
            new_state += [1, 0]
        elif action == 1:  # east
            new_state += [0, 1]
        elif action == 2:  # north
            new_state += [-1, 0]
        elif action == 3:  # west
            new_state += [0, -1]

        if new_state[0] < 6 and new_state[0] >= 0:
            if new_state[1] < 9 and new_state[1] >= 0:
                # Check for wall
                if (new_state[0], new_state[1]) not in self.walls:
                    current_state = copy.deepcopy(new_state)

        if self.current_state[0] == self.goal[0] and self.current_state[1] == self.goal[1]:
            reward = 1
            is_terminal = True
        else:
            reward = 0
            is_terminal = False
        return reward, self.current_state, is_terminal

    def env_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: response based on in_message
        This function is complete. You do not need to add code here.
        """
        pass
