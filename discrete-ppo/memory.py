import numpy as np


class ReplayBuffer:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.memory = {
            "states": [],
            "probs": [],
            "values": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }

    def sample(self):
        n_states = len(self.memory["states"])
        batch_starts = np.arange(0, n_states, self.batch_size)
        ids = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(ids)
        batches = [ids[i : i + self.batch_size] for i in batch_starts]
        return ({k: np.array(v) for k, v in self.memory.items()}, batches)

    def store_transition(self, state, prob, value, action, reward, done):
        self.memory["states"].append(state)
        self.memory["probs"].append(prob)
        self.memory["values"].append(value)
        self.memory["actions"].append(action)
        self.memory["rewards"].append(reward)
        self.memory["dones"].append(done)

    def clear_memory(self):
        for key in self.memory:
            self.memory[key] = []
