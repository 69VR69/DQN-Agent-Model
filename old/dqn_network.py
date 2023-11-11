import torch
import torch.nn as nn
import CONSTANTS as C

# Define a simple Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, C.SPACE_SIZE)
        self.fc2 = nn.Linear(C.SPACE_SIZE, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    #add a summary method to print out the network architecture
    def summary(self):
        print(self)