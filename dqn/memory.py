import numpy as np


class ReplayBuffer:
    def __init__(self, input_shape, n_actions, buffer_size=int(1e6)):
        self.buffer_size = int(buffer_size)
        self.states = np.zeros((self.buffer_size, *input_shape))
        self.next_states = np.zeros((self.buffer_size, *input_shape))
        self.actions = np.zeros((self.buffer_size, n_actions))
        self.rewards = np.zeros((self.buffer_size))
        self.dones = np.zeros((self.buffer_size), dtype=bool)
        self.mem_counter = 0

    def store_transition(self, state, action, reward, next_state, done):
        i = self.mem_counter % self.buffer_size
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.done[i] = done
        self.mem_counter += 1

    def generate_batches(self):
        pass