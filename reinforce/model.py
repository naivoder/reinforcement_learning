import torch
from torch import nn


class LanderNN(nn.Module):
    """
    Neural network model for Lunar Lander implemented using PyTorch.

    This model is a simple feedforward neural network with two hidden layers,
    designed to determine actions for the Lunar Lander game based on its state.

    Parameters
    ----------
    input_dims : tuple
        Dimensions of the input state. For Lunar Lander, this is typically the number
        of environment state variables such as position, velocity, angle, etc.
    n_actions : int
        Number of possible actions the agent can take.
    lr : float
        Learning rate for the optimizer.

    Attributes
    ----------
    device : str
        The device (CPU or GPU) on which the model will be trained.
    input_dims : tuple
        Input dimensions for the model.
    n_actions : int
        Number of actions the model can output.
    lr : float
        Learning rate used by the optimizer.
    model : torch.nn.Sequential
        The sequential model consisting of two hidden layers and an output layer.
    optimizer : torch.optim.Optimizer
        The optimizer for training the model, specifically Adam in this case.

    Examples
    --------
    >>> model = LanderNN(input_dims=(8,), n_actions=4, lr=0.001)
    >>> print(model)
    """

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
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing the state.

        Returns
        -------
        torch.Tensor
            The tensor containing the logits for each action.
        """
        return self.model(x)


if __name__ == "__main__":
    model = LanderNN(input_dims=(8), n_actions=4, lr=0.0005)
