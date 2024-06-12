import os
import torch


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
