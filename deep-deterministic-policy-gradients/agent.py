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
        super(DDPGAgent, self).__init__()
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

    def learn(self):
        if self.memory.mem_counter() < self.batch_size:
            return

        states, actions, rewards, next_states, done = self.replay_buffer.sample(
            self.batch_size
        )
        states = torch.tensor(states, torch.float).to(self.actor.device)
        actions = torch.tensor(actions, torch.float).to(self.actor.device)
        next_states = torch.tensor(next_states, torch.float).to(self.actor.device)
        rewards = torch.tensor(rewards, torch.float).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)

        target_actions = self.target_actor(next_states)
        target_critic_values = self.target_critic(next_states, target_actions)
        critic_values = self.critic(states, actions)

        # set target critic value to zero for terminal states
        target_critic_values[done] = 0.0  # fix dim issue
        target_critic_values = target_critic_values.view(-1)

        target = rewards + self.gamma * target_critic_values
        target = target.view(self.batch_size, 1)  # add batch dim

        self.critic.optimizer.zero_grad()
        critic_loss = torch.nn.functional.mse_loss(target, critic_values)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic(states, self.actor(states))
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

        return actor_loss, critic_loss
