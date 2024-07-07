import torch
import numpy as np
from memory import ReplayBuffer


class DeepQNetwork(torch.nn.Module):
    def __init__(self, input_shape, n_actions, alpha=3e-4, chkpt_file="weights/dqn.pt"):
        super(DeepQNetwork, self).__init__()
        self.chkpt_file = chkpt_file

        self.conv1 = torch.nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1_input_dim = self._calculate_fc1_input_dim(input_shape)
        self.fc1 = torch.nn.Linear(self.fc1_input_dim, 512)
        self.out = torch.nn.Linear(512, n_actions)

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=alpha)
        self.loss = torch.nn.MSELoss()  # use squared l1 instead of mse?

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = x.view(x.size()[0], -1)
        x = torch.nn.functional.relu(self.fc1(x))
        return self.out(x)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))

    def _calculate_fc1_input_dim(self, input_shape):
        dummy_input = torch.zeros(1, *input_shape)
        x = torch.nn.functional.relu(self.conv1(dummy_input))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        return x.numel()


class DDQNAgent:
    def __init__(
        self,
        env_name,
        input_shape,
        n_actions,
        alpha=3e-4,
        gamma=0.99,
        eps_min=0.01,
        eps_dec=5e-7,
        batch_size=64,
        mem_size=100000,
        replace_target_count=1000,
    ):
        self.gamma = gamma
        self.epsilon = 1.0
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.replace_target_count = replace_target_count
        self.counter = 0

        self.memory = ReplayBuffer(input_shape, int(mem_size), batch_size)
        self.q1 = DeepQNetwork(
            input_shape, n_actions, alpha, f"weights/{env_name}_q1.pt"
        )
        self.q2 = DeepQNetwork(
            input_shape, n_actions, alpha, f"weights/{env_name}_q2.pt"
        )

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.q.device)
            actions = self.q1(state)
            return torch.argmax(actions).item()

        return np.random.randint(0, self.n_actions)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        if self.counter % self.replace_target_count == 0:
            self.update_target_parameters()

        states, actions, rewards, next_states, dones = self.memory.sample()

        states = torch.FloatTensor(states).to(self.q1.device)
        actions = torch.IntTensor(actions).to(self.q1.device)
        next_states = torch.FloatTensor(next_states).to(self.q1.device)
        rewards = torch.FloatTensor(rewards).to(self.q1.device)
        dones = torch.BoolTensor(dones).to(self.q1.device)

        self.q1.optimizer.zero_grad()

        ids = np.arange(self.batch_size)
        q_pred = self.q1(states)[ids, actions]

        # get offline value of actions chosen by online policy
        next_actions = torch.argmax(self.q1(next_states), dim=1)
        target_vals = self.q2(next_states)[ids, next_actions]
        target_vals[dones] = 0.0

        q_target = rewards + self.gamma * target_vals

        loss = self.q1.loss(q_target, q_pred).to(self.q1.device)
        loss.backward()

        self.q1.optimizer.step()

        self.counter += 1
        self.decrement_epsilon()

    def decrement_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon - self.eps_dec)

    def update_target_parameters(self):
        self.q2.load_state_dict(dict(self.q1.named_parameters()))

    def save_checkpoint(self):
        self.q1.save_checkpoint()
        self.q2.save_checkpoint()

    def load_checkpoint(self):
        self.q1.load_checkpoint()
        self.q2.load_checkpoint()
