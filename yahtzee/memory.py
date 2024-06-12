import numpy as np


class ReplayBuffer:
    def __init__(self, input_shape, n_actions, size=100000):
        self.size = size
        self.counter = 0
        self.state_memory = np.zeros((self.size, *input_shape), dtype=np.float32)
        self.next_state_memory = np.zeros((self.size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.size, dtype=np.int64)
        self.reward_memory = np.zeros(self.size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.counter % self.size
        self.state_memory[index] = state
        self.next_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.counter, self.size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal
