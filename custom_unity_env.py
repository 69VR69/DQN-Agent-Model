import gymnasium
from gymnasium import spaces
import numpy as np
from environment import Environment


class CustomUnityEnv(gymnasium.Env):
    def __init__(self):
        super(CustomUnityEnv, self).__init__()

        # Initialize the socket communication manager
        self.env = Environment()
        self.env.start()

        # Define your observation and action spaces
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(27,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(8)  # Assuming 8 possible actions

        # Initialize variables
        self.state = None
        self.reward = None
        self.done = False

    def step(self, action):
        # Send the action to Unity
        self.reward, self.state, self.done = self.set_action(action)

        # Return observation, reward, done, and info
        return self.state, self.reward, self.done, False, {}

    def reset(self, seed=None):
        super().reset(seed=seed)
        # Reset the Unity environment and return the initial observation
        self.state = self.env.reset()
        self.done = False
        return self.state, {}

    def close(self):
        self.env.close()

    def render(self, mode="human"):
        # Implement rendering if needed (e.g., for visual observation)
        # Print a text-based representation of the environment state
        if self.state is not None:
            for row in self.state:
                print(" ".join([f"{val:.2f}" for val in row]))
            print(f"Reward: {self.reward}")
            print(f"Done: {self.done}")
        else:
            super(CustomUnityEnv, self).render(mode=mode)

    def set_action(self, action):
        # Use your server manager to send actions to Unity and get new state and reward
        return self.env.set_action(action)


# Example usage of your custom Unity environment
# if __name__ == "__main__":
#     env = CustomUnityEnv()

#     for episode in range(num_episodes):
#         obs = env.reset()
#         done = False
#         while not done:
#             action = env.action_space.sample()  # Replace with your RL agent's action
#             obs, reward, done, info = env.step(action)

#     env.close()
