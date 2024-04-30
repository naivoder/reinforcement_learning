import torch
from torch import nn


class LanderNN(nn.Module):
    def __init__(self, input_dims, n_actions, lr):
        super(LanderNN, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.lr = lr
        self.model = nn.Sequential(
            nn.Linear(*self.input_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_actions),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = LanderNN(input_dims=(8), n_actions=4, lr=0.0005)
