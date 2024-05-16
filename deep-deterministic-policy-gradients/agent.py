import torch
import numpy as np


class DDPGAgent(torch.nn.Module):
    def __init__(self, actor, critic, replay_buffer, action_noise):
        super(self, DDPGAgent).__init__()
        self.actor = actor
        self.critic = critic
        self.target_actor = actor
        self.target_critic = critic
        self.replay_buffer = replay_buffer
        self.action_noise = action_noise

    def choose_action(self, state):
        pass

    def store_transition(self, state, action, reward, next_state, done):
        pass

    def save_checkpoints(self):
        pass

    def load_checkpoints(self):
        pass
