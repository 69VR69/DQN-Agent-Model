from environment import Environment
from batch import Batch, Experience
from DQNModel import Model

class Collector:
    def __init__(self, batch_size, model = Model()):
        self.batch_size = batch_size
        self.batch = Batch(batch_size)
        self.model = model
    
    def fill_batch(self, env):
        while not self.batch.is_full():
            self.batch.add(self.collect_experience(env))
    
    def collect_experience(self, env):
        if not env.is_running():
            env.start()
            env.reset()
        
        # get action
        current_state = env.get_state()
        action = self.model.get_action(current_state)

        # send action to server
        reward, next_state, done = env.send_action(action)

        return Experience(current_state, action, reward, next_state, done)
    
    # to string
    def __str__(self):
        return "Collector:\n" + str(self.batch)


# Test
env = Environment()  
collector = Collector(10)
collector.fill_batch(env)
print(collector)