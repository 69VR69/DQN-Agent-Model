import random

class Model:
    def __init__(self, epsilon = 0.99, num_actions = 9):
        self.epsilon = epsilon
        self.num_actions = num_actions

    def get_action(self, state):
        # get a random number between 0 and 1
        random_number = random.random()
        # if number < epsilon, return random action else return best action
        if random_number < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return self.get_best_action(state)
        
    def get_best_action(self, state):
        return 0 # TODO call the model to get the best action for the given state
