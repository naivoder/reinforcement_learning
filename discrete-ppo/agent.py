import torch
from networks import Actor, Critic
from memory import ReplayBuffer
import numpy as np


class DiscretePPOAgent:
    def __init__(
        self,
        input_dims,
        n_actions,
        gamma=0.99,
        alpha=3e-4,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10,
    ):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = 1e-3
        self.actor = Actor(input_dims, n_actions, alpha)
        self.critic = Critic(input_dims, alpha)
        self.memory = ReplayBuffer(batch_size)

    def remember(self, state, state_, action, probs, reward, done):
        self.memory.store_transition(state, state_, action, probs, reward, done)

    def save_checkpoints(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def save_checkpoints(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state), dtype=torch.float).to(
                self.actor.device
            )

            dist = self.actor(state)
            action = dist.sample()
            probs = dist.log_prob(action)

        return (
            action.cpu().numpy().flatten().item(),
            probs.cpu().numpy().flatten().item(),
        )

    def calc_adv_and_returns(self, memories):
        states, new_states, r, dones = memories
        with torch.no_grad():
            values = self.critic(states)
            values_ = self.critic(new_states)
            deltas = r + self.gamma * values_ - values
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]
            for dlt, mask in zip(deltas[::-1], dones[::-1]):
                advantage = dlt + self.gamma * self.gae_lambda * adv[-1] * (1 - mask)
                adv.append(advantage)
            adv.reverse()
            adv = adv[:-1]
            adv = torch.tensor(adv).float().unsqueeze(1).to(self.critic.device)
            returns = adv + values
            adv = (adv - adv.mean()) / (adv.std() + 1e-4)
        return adv, returns

    def learn(self):
        state_arr, new_state_arr, action_arr, old_prob_arr, reward_arr, dones_arr = (
            self.memory.sample()
        )
        state_arr = torch.tensor(state_arr, dtype=torch.float).to(self.critic.device)
        action_arr = torch.tensor(action_arr, dtype=torch.float).to(self.critic.device)
        old_prob_arr = torch.tensor(old_prob_arr, dtype=torch.float).to(
            self.critic.device
        )
        new_state_arr = torch.tensor(new_state_arr, dtype=torch.float).to(
            self.critic.device
        )
        r = (
            torch.tensor(reward_arr, dtype=torch.float)
            .unsqueeze(1)
            .to(self.critic.device)
        )
        adv, returns = self.calc_adv_and_returns(
            (state_arr, new_state_arr, r, dones_arr)
        )
        for epoch in range(self.n_epochs):
            batches = self.memory.generate_batches()
            for batch in batches:
                states = state_arr[batch]
                old_probs = old_prob_arr[batch]
                actions = action_arr[batch]

                dist = self.actor(states)
                new_probs = dist.log_prob(actions)
                prob_ratio = torch.exp(
                    new_probs.sum(-1, keepdim=True) - old_probs.sum(-1, keepdim=True)
                )
                # print("probs ratio", prob_ratio.shape)
                weighted_probs = adv[batch] * prob_ratio
                weighted_clipped_probs = (
                    torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * adv[batch]
                )
                # print("weighted clipped probs", weighted_clipped_probs.shape)
                entropy = dist.entropy().sum(-1, keepdims=True)
                # print("entropy", entropy.shape)
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs)
                actor_loss -= self.entropy_coefficient * entropy
                # print("actor loss", actor_loss.shape)

                self.actor.optimizer.zero_grad()
                actor_loss.mean().backward()
                self.actor.optimizer.step()

                critic_value = self.critic(states)
                critic_loss = (critic_value - returns[batch]).pow(2).mean()
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

        self.memory.clear_memory()
