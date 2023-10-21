import torch

class TrainingLoop:
    def __init__(self, agent, env_interface, num_episodes, batch_size, gamma):
        self.agent = agent
        self.env_interface = env_interface
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.gamma = gamma

    def train_agent(self):
        for episode in range(self.num_episodes):
            state = self.env_interface.reset()
            done = False
            total_reward = 0

            while not done:
                epsilon = max(0.1, 1.0 - episode / 500)
                state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
                action = self.agent.select_action(state_tensor, epsilon)
                next_state, reward, done = self.env_interface.set_action(action)
                self.agent.store_experience(state_tensor, action, reward, torch.tensor(next_state, dtype=torch.float).unsqueeze(0), done)
                state = next_state
                total_reward += reward

                self.agent.train(self.batch_size, self.gamma)

            print(f"Episode {episode}: Total Reward: {total_reward}\n")

        self.agent.save_model("dqn_model.pth")

    def save_trained_model(self, filename):
        self.agent.save_model(filename)

    def load_trained_model(self, filename):
        # Load a pre-trained DQN model
        return torch.load(filename)