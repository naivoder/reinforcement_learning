from model import LanderNN
import numpy as np
import torch
import torch.functional as F


class Agent:
    def __init__(self, lr, input_dims, n_actions=4, gamma=0.99):
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma

        self.reward_memory = []
        self.action_memory = []

        self.policy = LanderNN(
            input_dims=self.input_dims, n_actions=self.n_actions, lr=self.lr
        )

    def choose_action(self, state):
        # need to add batch dimension to state so pytorch doesn't freak out
        state = torch.Tensor([state]).to(self.policy.device)
        probs = F.softmax(self.policy(state))

        # build distribution over probabilities
        action_probs = torch.distributions.Categorical(probs)

        # sample action from distribution
        action = action_probs.sample()

        # calculate log prob of action
        log_prob = action_probs.log_prob(action)
        self.action_memory.append(log_prob)

        # dereference pytorch tensor (gym doesn't like torch tensors)
        return action.item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad()
        Gt = np.zeros_like(self.reward_memory, dtype=np.float64)

        for i in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for j in range(i, len(self.reward_memory)):
                G_sum += discount * self.reward_memory[j]
                discount *= self.gamma
            Gt[i] = G_sum

        Gt = torch.Tensor(Gt).to(self.policy.device)

        loss = 0
        for g, logprob in zip(Gt, self.action_memory):
            loss += -g * logprob
        loss.backwards()
        self.policy.optimizer.step()

        self.reward_memory = []
        self.action_memory = []


if __name__ == "__main__":
    agent = Agent(0.0005, (8))
