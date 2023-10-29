from neural_network import QNetwork
import random
import torch
from torch.optim import Adam


class Model:
    def __init__(self, epsilon=0.99, gamma = 0.99, num_state=27, num_actions=9, learning_rate=0.001):
        self.epsilon = epsilon
        self.gamma = gamma
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
        return self.q_network.forward(state).argmax().item()

    def get_loss(self, states, actions, q_targets):
        # get the loss for the state, action and q_target
        predict = self.q_network.forward(states)
        return torch.mean(
            torch.mul(torch.square(torch.sub(predict, q_targets)), actions)
        )

    def get_q_target(self, next_states, gammas, rewards):
        predict = self.q_network.forward(next_states).max(1).values
        gammas = gammas[None, :]
        print(" shape of predict", predict.shape)
        print(" shape of gammas", gammas.shape)
        predict_masked = torch.mm(gammas, predict)
        print("predict value", predict)
        print("rewards value", rewards)
        Qtargets = torch.tensor(predict_masked.add(rewards), requires_grad=True)
        return Qtargets

    def get_gamma(self, dones):
        return torch.mul(torch.tensor(self.gamma), torch.logical_not(dones))