import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state (37)
            action_size (int): Dimension of each action (4)
            fc1_units (int): Number of nodes in first hidden layer (64)
            fc2_units (int): Number of nodes in second hidden layer (64)
        """
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(random.randint(0, 1000))
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units

        # Q-Network
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)