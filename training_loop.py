from collector import Collector
from model import Model
from environment import Environment

class TrainingLoop:
    def __init__(self, batch_size, num_episodes, model = Model()):
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.model = model
        self.collector = Collector(batch_size, model)
        self.env = Environment()

    def run(self):
        for i in range(self.num_episodes):
            self.collector.fill_batch(self.env)
            batch = self.collector.get_batch()
            self.collector.flush_batch()
            self.train(batch)

    def train(self, batch):
        states = batch.get_states()
        actions = batch.get_actions()
        rewards = batch.get_rewards()
        next_states = batch.get_next_states()
        dones = batch.get_dones()

        gammas = self.model.get_gamma(dones)
        q_targets = self.model.get_q_target(states, gammas, rewards)
        loss = self.model.get_loss(states, actions, q_targets)

        self.model.optimiser.zero_grad()
        loss.backward()
        self.model.optimiser.step()

        self.model.epsilon = self.model.epsilon * 0.99

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