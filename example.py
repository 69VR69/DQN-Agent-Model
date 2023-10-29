import torch
import torch.nn as nn
import torch.optim as optim
import random

# Define the environment and agent
class Environment:
    def __init__(self):
        self.state_size = 26
        self.action_size = 3

    def step(self, action):
        # Simulate the environment step and return a reward
        return random.uniform(0, 1)

class Agent:
    def __init__(self, state_size, action_size):
        self.model = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 2)
        else:
            with torch.no_grad():
                q_values = self.model(torch.tensor(state, dtype=torch.float32))
                return q_values.argmax().item()

def mean(array):
    if len(array) == 0:
        return None
    return sum(array) / len(array)

def train_model(agent, states, actions, rewards, next_states):
    batch_size = 32
    size = len(next_states)
    losses = []

    for b in range(0, size, batch_size):
        to = b + batch_size if b + batch_size < size else size
        states_batch = torch.tensor(states[b:to], dtype=torch.float32)
        actions_batch = torch.tensor(actions[b:to], dtype=torch.float32)
        rewards_batch = torch.tensor(rewards[b:to], dtype=torch.float32)
        next_states_batch = torch.tensor(next_states[b:to], dtype=torch.float32)

        q_targets = next_states_batch.max(dim=1, keepdim=True)[0] * 0.99 + rewards_batch

        def model_loss():
            q_values = agent.model(states_batch)
            loss = ((q_values - q_targets) ** 2 * actions_batch).mean()
            return loss

        agent.optimizer.zero_grad()
        loss = model_loss()
        loss.backward()
        agent.optimizer.step()
        losses.append(loss.item())

    print("Mean loss", mean(losses))

# Main training loop
env = Environment()
agent = Agent(env.state_size, env.action_size)

eps = 1.0
states = []
rewards = []
reward_mean = []
next_states = []
actions = []

state = [0] * env.state_size

for epi in range(150):
    reward = 0
    step = 0
    while step < 400:
        action = agent.select_action(state, eps)
        reward = env.step(action)
        state2 = [0] * env.state_size
        state2[action] = 1

        index = random.randint(0, len(states))
        states.insert(index, state)
        rewards.insert(index, [reward])
        reward_mean.insert(index, reward)
        next_states.insert(index, state2)
        actions.insert(index, state2)

        if len(states) > 10000:
            states.pop(0)
            rewards.pop(0)
            reward_mean.pop(0)
            next_states.pop(0)
            actions.pop(0)

        state = state2
        step += 1

    eps = max(0.1, eps * 0.99)

    if epi % 5 == 0:
        print("---------------")
        print("rewards mean", mean(reward_mean))
        print("episode", epi)
        train_model(agent, states, actions, rewards, next_states)

print("Training complete")