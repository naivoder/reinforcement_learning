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
