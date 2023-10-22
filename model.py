from neural_network import QNetwork
import random
import torch
from torch.optim import Adam

class Model:
    def __init__(self, epsilon = 0.99, num_state = 27,num_actions = 9, learning_rate = 0.001):
        self.epsilon = epsilon
        self.num_state = num_state
        self.num_actions = num_actions
        self.q_network = QNetwork(num_state, num_actions)
        self.optimiser = Adam(self.q_network.parameters(), lr=learning_rate)

    def get_action(self, state):
        # get a random number between 0 and 1
        random_number = random.random()
        # if number < epsilon, return random action else return best action
        if random_number < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return self.get_best_action(state)
        
    def get_best_action(self, state):
        # get the best action for the state
        return  self.q_network.forward(state).argmax().item()
    
    def get_loss(self, states, actions, q_targets):
        # get the loss for the state, action and q_target
        predict = self.q_network.forward(states)
        return predict.sub(q_targets).square().mul(actions).mean()
    
    def get_q_target(self, states, gammas, rewards):
        predict = self.q_network.forward(states).max(1)
        predict_masked = predict.mul(gammas).add(rewards)
        Qtargets = torch.tensor(predict_masked.buffer().values, shape=[len(states), 1])
        return Qtargets
    
# Test
model = Model()
# test get q target
states = torch.randn(10, 27)
gammas = torch.randn(10, 1)
rewards = torch.randn(10, 1)
q_targets = model.get_q_target(states, gammas, rewards)
print(q_targets)
# test get loss
states = torch.randn(10, 27)
actions = torch.randn(10, 9)
loss = model.get_loss(states, actions, q_targets)
print(loss)