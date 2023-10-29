import numpy as np
import torch
from collector import Collector
from model import Model
from environment import Environment

class TrainingLoop:
    def __init__(self, batch_size, num_episodes, model = Model()):
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.model = model
        self.env = Environment()
        self.collector = Collector(self.batch_size, self.model, self.env)
        self.losses = []
        self.rewards = []

    def run(self):
        for i in range(self.num_episodes):
            batch = self.collector.get_batch_from_env()
            print("Batch collected :", batch)
            self.train(batch)
            if(i % 5 == 0):
                print("Episode: " + str(i) + "=>\n\tmean loss: " + str(np.mean(self.losses)) + "\n\tmean reward: " + str(np.mean(self.rewards)))

    def train(self, batch):
        states = torch.tensor(batch.get_states(), dtype=torch.float32)
        actions = torch.tensor(batch.get_actions(), dtype=torch.float32)
        rewards = torch.tensor(batch.get_rewards(), dtype=torch.float32)
        next_states = torch.tensor(batch.get_next_states(), dtype=torch.float32)
        dones = torch.tensor(batch.get_dones(), dtype=torch.bool)

        gammas = self.model.get_gamma(dones)
        q_targets = self.model.get_q_target(next_states, gammas, rewards)

        self.model.optimiser.zero_grad()
        loss = self.model.get_loss(states, actions, q_targets)
        loss.backward()
        self.model.optimiser.step()

        # For logging
        self.rewards.append(rewards)
        self.losses.append(loss)

        self.model.epsilon = max(0.1, self.model.epsilon * 0.99)

    def get_model(self):
        return self.model

    def __str__(self):
        return "TrainingLoop:\n" + str(self.model)
    
# Test
training_loop = TrainingLoop(10, 100)
training_loop.run()
print(training_loop)

# Expected output
# TrainingLoop:
# Model:
# QNetwork(
#   (fc1): Linear(in_features=27, out_features=64, bias=True)
#   (fc2): Linear(in_features=64, out_features=64, bias=True)
#   (fc3): Linear(in_features=64, out_features=9, bias=True)
# )
# Optimiser: Adam (
# Parameter Group 0
#     amsgrad: False
#     betas: (0.9, 0.999)
#     eps: 1e-08
#     lr: 0.001
#     weight_decay: 0
# )
# Epsilon: 0.3660323412732292
# Gamma: 0.99
# Loss: tensor(0.0001, grad_fn=<MeanBackward0>)