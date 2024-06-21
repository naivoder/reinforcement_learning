import torch
import numpy as np
import networks
import memory


class Agent(torch.nn.Module):
    def __init__(
        self,
        input_dims,
        n_actions,
        tau,
        alpha,
        gamma=0.99,
        h1_size=400,
        h2_size=300,
        mem_size=int(1e6),
    ):
        super(Agent, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.tau = tau
        self.alpha = alpha
        self.gamma = gamma
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.mem_size = mem_size

        self.memory = memory.ReplayBuffer(
            self.input_dims, self.n_actions, self.mem_size
        )

        self.Q1 = networks.CriticNetwork(
            self.input_dims,
            self.n_actions,
            self.h1_size,
            self.h2_size,
            chkpt_path="weights/critic_1.pt",
        )
        self.Q2 = networks.CriticNetwork(
            self.input_dims,
            self.n_actions,
            self.h1_size,
            self.h2_size,
            chkpt_path="weights/critic_2.pt",
        )
        self.V = networks.ValueNetwork(
            self.input_dims, self.h1_size, self.h2_size, chkpt_path="weights/value.pt"
        )
        self.V_target = networks.ValueNetwork(
            self.input_dims,
            self.h1_size,
            self.h2_size,
            chkpt_path="weights/value_target.pt",
        )
        self.Actor = networks.ActorNetwork(
            self.input_dims, self.n_actions, self.h1_size, self.h2_size, max_action=None
        )

        self.update_network_params()

    def choose_action(self, state):
        state = torch.tensor(state, torch.float32)
        return self.Actor(state)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def update_network_params(self, tau=None):
        if tau is None:
            tau = self.tau

        value_params = dict(self.V.named_parameters())
        target_value_params = dict(self.V_target.named_parameters())
        for name in value_params:
            value_params[name] = (
                tau * value_params[name].clone()
                + (1 - tau) * target_value_params[name].clone()
            )
        self.V_target.load_state_dict(value_params)

    def save_model(self):
        self.V.save_checkpoint()
        self.V_target.save_checkpoint()
        self.Q1.save_checkpoint()
        self.Q2.save_checkpoint()
        self.Actor.save_checkpoint()

    def load_model(self):
        self.V.load_checkpoint()
        self.V_target.load_checkpoint()
        self.Q1.load_checkpoint()
        self.Q2.load_checkpoint()
        self.Actor.load_checkpoint()
