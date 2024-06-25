import numpy as np
import torch as T
from memory import ReplayBuffer
import torch
import os


class DuelingDeepQNetwork(torch.nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir="weights"):
        super(DuelingDeepQNetwork, self).__init__()

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc_1 = torch.nn.Linear(*input_dims, 1024)
        self.ln_1 = torch.nn.LayerNorm(1024)
        self.fc_2 = torch.nn.Linear(1024, 512)
        self.ln_2 = torch.nn.LayerNorm(512)
        self.V = torch.nn.Linear(512, 1)
        self.A = torch.nn.Linear(512, n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = torch.nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = torch.nn.functional.relu(self.ln_1(self.fc_1(state)))
        x = torch.nn.functional.relu(self.ln_2(self.fc_2(x)))

        V = self.V(x)
        A = self.A(x)

        return V, A

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class DuelingDDQNAgent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_dims,
        batch_size,
        action_space_shape,
        n_actions=44,
        eps_min=0.01,
        eps_dec=5e-7,
        replace=1000,
        chkpt_dir="weights/ddqn",
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.action_space_shape = action_space_shape
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.chkpt_dir = chkpt_dir
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(input_dims, self.n_actions)

        self.q_eval = DuelingDeepQNetwork(
            self.lr,
            self.n_actions,
            input_dims=self.input_dims,
            name="q_eval",
            chkpt_dir=self.chkpt_dir,
        )
        self.q_next = DuelingDeepQNetwork(
            self.lr,
            self.n_actions,
            input_dims=self.input_dims,
            name="q_next",
            chkpt_dir=self.chkpt_dir,
        )

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def choose_action(self, observation, valid_categories):
        remaining_rolls = observation[5]

        if np.random.random() > self.epsilon:
            state = np.array([observation], copy=False, dtype=np.float32)
            state_tensor = T.tensor(state).to(self.q_eval.device)
            _, advantages = self.q_eval.forward(state_tensor)
            advantages = advantages.squeeze()  # Ensure the tensor shape is correct

            if remaining_rolls > 0:
                # Choose a re-roll action
                action = T.argmax(advantages[:31]).item()
            else:
                # Choose a score action from valid categories
                valid_advantages = [
                    advantages[31 + cat].item() for cat in valid_categories
                ]
                action = 31 + valid_categories[np.argmax(valid_advantages)]
        else:
            if remaining_rolls > 0:
                # Randomly choose a re-roll action
                action = np.random.choice(31)
            else:
                # Randomly choose a score action from valid categories
                action = 31 + np.random.choice(valid_categories)

        return action

    def replace_target_network(self):
        if (
            self.replace_target_cnt is not None
            and self.learn_step_counter % self.replace_target_cnt == 0
        ):
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        )

    def learn(self):
        if self.memory.counter < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]

        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()
