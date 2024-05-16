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
        state = torch.Tensor(np.array(state)).to(self.actor.device)
        actions = self.actor(state)
        action_probs = torch.distributions.categorical(actions)
        action = action_probs.sample()
        return action.item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.store_transition(state, action, reward, next_state, done)

    def save_checkpoints(self, epoch, loss):
        self.actor.save_checkpoint(epoch, loss)
        self.critic.save_checkpoint(epoch, loss)

    def load_checkpoints(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
