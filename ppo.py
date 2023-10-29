import gym
import numpy as np
from stable_baselines3 import PPO
from custom_unity_env import CustomUnityEnv
from stable_baselines3.common.env_checker import check_env
from save_callback import SaveCallback

# Create your custom Unity environment
env = CustomUnityEnv()

# Define the RL agent
model = PPO("MlpPolicy", env, verbose=2)

# Train the agent
model.learn(
    total_timesteps=10000,
    callback=SaveCallback("models", 1000),
    log_interval=5,
    progress_bar=True,
)

# Test the trained agent
obs = env.reset()[0]
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    if done:
        obs = env.reset()
