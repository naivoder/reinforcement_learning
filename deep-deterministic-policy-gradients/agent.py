import torch
import numpy as np
from noise import OrnsteinUhlenbeckActionNoise
from actor import ActorNetwork
from critic import CriticNetwork
from memory import ReplayBuffer


class DDPGAgent(torch.nn.Module):
    def __init__(
        self,
        input_dims,
        n_actions,
        tau,
        alpha=1e-4,
        beta=1e-3,
        gamma=0.99,
        batch_size=64,
        mem_size=1e6,
    ):
        super(self, DDPGAgent).__init__()
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.batch_size = batch_size

        self.replay_buffer = ReplayBuffer(mem_size, input_dims, n_actions)
        self.action_noise = OrnsteinUhlenbeckActionNoise(np.zeros(input_dims))

        self.actor = ActorNetwork(alpha, input_dims, n_actions)
        self.target_actor = ActorNetwork(alpha, input_dims, n_actions)

        self.critic = CriticNetwork(beta, input_dims, n_actions)
        self.target_critic = CriticNetwork(beta, input_dims, n_actions)

        self.update_network_parameters(tau=1)

    def choose_action(self, state):
        self.actor.eval()
        state = torch.Tensor(np.array(state), dtype=torch.float).to(self.actor.device)
        mu = self.actor(state).to(self.actor.device)
        # add noise to deterministic output
        mu = mu + torch.tensor(self.noise(), dtype=torch.float()).to(self.actor.device)
        self.actor.train()
        return mu.cpu().detach().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.store_transition(state, action, reward, next_state, done)

    def save_checkpoints(self, epoch, loss):
        self.actor.save_checkpoint(epoch, loss)
        self.target_actor.save_checkpoint(epoch, loss)
        self.critic.save_checkpoint(epoch, loss)
        self.target_critic.save_checkpoint(epoch, loss)

    def load_checkpoints(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()
