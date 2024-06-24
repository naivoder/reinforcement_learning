import torch
from networks import Actor, Critic
from memory import ReplayBuffer
import numpy as np
from tqdm import tqdm


class DiscretePPOAgent(torch.nn.Module):
    def __init__(
        self,
        input_dims,
        n_actions,
        alpha=3e-4,
        lamda=0.95,
        gamma=0.99,
        clip=0.2,
        batch_size=64,
        N=2048,
        epochs=10,
    ):
        super(DiscretePPOAgent, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.alpha = alpha
        self.lamda = lamda
        self.gamma = gamma
        self.clip = clip
        self.batch_size = batch_size
        self.N = N
        self.epochs = epochs

        self.memory = ReplayBuffer(self.batch_size)
        self.actor = Actor(self.input_dims, self.n_actions, self.alpha)
        self.critic = Critic(self.input_dims, self.alpha)

    def choose_action(self, state):
        state = torch.FloatTensor([state]).to(self.actor.device)

        value = self.critic(state)
        dist = self.actor(state)
        action = dist.sample()
        probs = dist.log_prob(action)

        # to squeeze or not to squeeze, that is the question
        return action.squeeze().item(), probs.squeeze().item(), value.squeeze().item()

    def learn(self):
        for _ in tqdm(range(self.epochs)):
            states, old_probs, actions, values, rewards, dones = self.memory.sample()
            batches = self.memory.generate_batches()

            advantage = np.zeros(len(rewards), dtype=np.float32)

            for i in range(len(rewards) - 1):
                at, discount = 0, 1
                for j in range(i, len(rewards) - 1):
                    at += discount * (
                        rewards[j] + self.gamma * values[j + 1] * (1 - dones[j])
                    )
                    discount *= self.gamma * self.lamda
                advantage[i] = at

            advantage = torch.tensor(advantage).to(self.actor.device)
            values = torch.tensor(values).to(self.actor.device)

            for batch in batches:
                states_batch = torch.FloatTensor(states[batch]).to(self.actor.device)
                old_probs_batch = torch.tensor(old_probs[batch]).to(self.actor.device)
                actions_batch = torch.tensor(actions[batch]).to(self.actor.device)

                dist = self.actor(states_batch)
                critic_value = self.critic(states_batch)

                new_probs = dist.log_prob(actions_batch)
                ratio = (new_probs - old_probs_batch).exp()

                weighted_probs = advantage[batch] * ratio
                clipped_probs = (
                    torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage[batch]
                )

                actor_loss = -torch.min(weighted_probs, clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = torch.mean((returns - critic_value) ** 2)

                loss = actor_loss + 0.5 * critic_loss.mean()

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

            self.memory.clear_memory()

    def save_checkpoints(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_checkpoints(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
