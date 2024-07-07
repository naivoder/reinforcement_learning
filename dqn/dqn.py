import torch


class ActionValue(torch.nn.Module):
    def __init__(self, input_shape, n_actions, alpha=3e-4, chkpt_file="weights/dqn.pt"):
        super(ActionValue, self).__init__()
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


class DQNAgent:
    def __init__(
        self,
    ):
        pass

    def choose_action(self):
        pass

    def store_transition(
        self,
    ):
        pass

    def learn(
        self,
    ):
        pass

    def update_target_parameters(self):
        pass
