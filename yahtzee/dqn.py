import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from yahtzee import YahtzeeEnv, plot_scores


class DQN(nn.Module):
    def __init__(self, state_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 5),  # Output only for dice actions
        )

    def forward(self, x):
        return self.network(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, state_dim, memory_size, batch_size, device):
        self.device = torch.device(device)
        self.model = DQN(state_dim).to(self.device)
        self.target_model = DQN(state_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters())
        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.gamma = 0.99  # Discount factor

    def select_action(self, state, epsilon):
        state_tensor = self.prepare_state_tensor(state).to(self.device)
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.model(state_tensor)
                dice_action = (torch.sigmoid(q_values) > 0.5).int().squeeze().tolist()
        else:
            dice_action = np.random.choice([0, 1], size=5).tolist()
        return dice_action

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))

        state_batch = torch.stack(
            [self.prepare_state_tensor(state).to(self.device) for state in batch[0]]
        ).squeeze(1)
        # print("State Batch:", state_batch.shape)

        action_batch = torch.stack(
            [
                torch.tensor(action, dtype=torch.float32, device=self.device)
                for action in batch[1]
            ]
        )
        # print("Action Batch:", action_batch.shape)

        reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=self.device)
        # print("Reward Batch:", reward_batch.shape)

        next_state_batch = torch.stack(
            [self.prepare_state_tensor(state).to(self.device) for state in batch[3]]
        ).squeeze(1)
        # print("Next State Batch:", next_state_batch.shape)

        state_action_values = self.model(state_batch)
        # print("State Action Values:", state_action_values.shape)

        action_values = torch.sum(state_action_values * action_batch, dim=1)
        # print("Action Values:", action_values)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if next_state_batch.size(0) > 0:
            next_state_q_values = self.target_model(next_state_batch)
            max_next_state_values = next_state_q_values.max(1)[0].detach()
            if max_next_state_values.dim() > 1:
                max_next_state_values = max_next_state_values.squeeze()
            next_state_values[: max_next_state_values.size(0)] = max_next_state_values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = nn.SmoothL1Loss()(action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def prepare_state_tensor(self, state):
        dice = torch.tensor(state["dice"], dtype=torch.float32)
        scorecard = torch.tensor(state["scored_categories"], dtype=torch.float32)
        remaining_rolls = torch.tensor([state["remaining_rolls"]], dtype=torch.float32)
        state_tensor = torch.cat((dice, scorecard, remaining_rolls), dim=0).unsqueeze(0)
        return state_tensor

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())


def calculate_epsilon(
    episode,
    half_life,
    min_epsilon=0.01,
    max_epsilon=1.0,
):
    """
    Calculate the epsilon value for the given episode using exponential decay.

    Parameters:
        min_epsilon (float): The minimum value epsilon can decay to.
        max_epsilon (float): The maximum value of epsilon (starting value).
        episode (int): The current episode number.
        half_life (int): The number of episodes at which epsilon should be about half of max_epsilon.

    Returns:
        float: The epsilon value for the current episode.
    """
    decay_rate = np.log(2) / half_life
    epsilon = max_epsilon * np.exp(-decay_rate * episode)
    epsilon = max(epsilon, min_epsilon)

    return epsilon


if __name__ == "__main__":
    env = YahtzeeEnv()
    agent = Agent(
        19,
        memory_size=10000,
        batch_size=128,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    scores = []
    num_episodes = 50000
    epsilon = 1
    min_epsilon = 0.01
    running_avg = 0
    eps_decay = 0.9999

    for i_episode in range(num_episodes):
        print(
            f"Playing Episode: {i_episode} of {num_episodes}\t Running Avg: {running_avg}\t Epsilon: {epsilon:.4f}",
            end="\r",
        )
        state, _ = env.reset()
        done = False

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state

        agent.optimize_model()
        scores.append(env.get_total_score())

        if epsilon > 0.01:
            epsilon = max(0.01, epsilon * eps_decay)

        if i_episode != 0 and i_episode % 100 == 0:
            running_avg = np.mean(scores[-100:])
            agent.update_target_network()

        print(" " * 1000, end="\r")

    print(f"Episode {i_episode}: Total reward: {env.get_total_score()}")
    env.render_scorecard()
    plot_scores(scores)
