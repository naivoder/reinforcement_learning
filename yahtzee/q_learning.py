from yahtzee import YahtzeeEnv, plot_scores
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from itertools import count


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
            nn.Linear(128, 18),  # 5 for dice_action + 13 for score_action
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
                dice_action = torch.sigmoid(q_values[:, :5]) > 0.5
                score_action = q_values[:, 5:].max(1)[1].item()
                action = {
                    "dice_action": dice_action.numpy().astype(int)[0],
                    "score_action": score_action,
                }
        else:
            action = {
                "dice_action": np.random.choice([0, 1], size=5),
                "score_action": np.random.choice(range(13)),
            }
        return action

    def prepare_state_tensor(self, state):
        dice = torch.tensor(state["dice"], dtype=torch.float32)
        scorecard = torch.tensor(state["scorecard"], dtype=torch.float32)
        potential_scores = torch.tensor(state["potential_scores"], dtype=torch.float32)
        remaining_rolls = torch.tensor([state["remaining_rolls"]], dtype=torch.float32)
        state_tensor = torch.cat(
            (dice, scorecard, potential_scores, remaining_rolls), dim=0
        ).unsqueeze(0)
        return state_tensor

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))

        state_batch = torch.stack(
            [self.prepare_state_tensor(state).to(self.device) for state in batch[0]]
        ).squeeze(1)
        dice_action_batch = torch.tensor(
            np.array([action["dice_action"] for action in batch[1]], dtype=np.float32),
            device=self.device,
        )
        score_action_batch = torch.tensor(
            [action["score_action"] for action in batch[1]],
            dtype=torch.int64,
            device=self.device,
        )

        reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=self.device)
        next_state_batch = torch.stack(
            [
                self.prepare_state_tensor(state).to(self.device)
                for state in batch[3]
                if state is not None
            ]
        ).squeeze(1)
        non_final_mask = torch.tensor(
            [s is not None for s in batch[3]], dtype=torch.bool, device=self.device
        )

        state_action_values = self.model(state_batch)
        dice_values = torch.sum(dice_action_batch * state_action_values[:, :5], dim=1)
        score_values = (
            state_action_values[:, 5:]
            .gather(1, score_action_batch.unsqueeze(-1))
            .squeeze(-1)
        )
        combined_q_values = dice_values + score_values

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if len(next_state_batch) > 0:
            next_state_values[non_final_mask] = (
                self.target_model(next_state_batch).max(1)[0].detach()
            )

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = nn.SmoothL1Loss()(
            combined_q_values.unsqueeze(-1), expected_state_action_values.unsqueeze(-1)
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Updates the target network with the current weights of the model."""
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
        32,
        memory_size=10000,
        batch_size=128,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    scores = []
    num_episodes = 500000
    epsilon = 1
    min_epsilon = 0.01
    running_avg = 0
    half_life = num_episodes // 2

    for i_episode in range(num_episodes):
        print(
            f"Playing Episode: {i_episode} of {num_episodes}\t Running Avg: {running_avg}",
            end="\r",
        )
        state, _ = env.reset()
        done = False

        score = 0
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            score += reward

        agent.optimize_model()

        scores.append(score)

        epsilon = calculate_epsilon(i_episode, half_life)

        if i_episode != 0 and i_episode % 100 == 0:
            running_avg = np.mean(scores[-100:])
            agent.update_target_network()

        print(" " * 50, end="\r")

    print(f"Episode {i_episode}: Total reward: {env.get_total_score()}")
    env.render_scorecard()
    plot_scores(scores)
