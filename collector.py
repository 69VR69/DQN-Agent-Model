import torch
from environment import Environment
from batch import Batch, Experience
from model import Model


class Collector:
    def __init__(self, batch_size, model=Model(), env=Environment()):
        self.batch_size = batch_size
        self.batch = Batch(batch_size)
        self.model = model
        self.env = env

    def fill_batch(self):
        while not self.batch.is_full():
            self.batch.add(self.collect_experience())

    def get_batch_from_env(self):
        self.fill_batch()
        batch = self.get_batch()
        self.flush_batch()
        return batch

    def get_batch(self):
        return self.batch

    def flush_batch(self):
        self.batch = Batch(self.batch_size)

    def collect_experience(self):
        if not self.env.is_running():
            self.env.start()
            self.env.reset()

        # get action
        current_state = torch.tensor(self.env.get_state(), dtype=torch.float32)
        action = self.model.get_action(current_state)

        # send action to server
        reward, next_state, done = self.env.set_action(action)

        return Experience(current_state, action, reward, next_state, done)

    # to string
    def __str__(self):
        return "Collector:\n" + str(self.batch)


# # Test
# env = Environment()
# collector = Collector(10)
# collector.fill_batch(env)
# print(collector)
