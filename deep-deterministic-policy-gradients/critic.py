import torch
import numpy as np


class CriticNetwork(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        n_actions,
        h1_size=400,
        h2_size=300,
        lr=1e-3,
        decay=1e-2,
        chkpt_path="critic.pt",
    ):
        super(CriticNetwork, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.lr = lr
        self.decay = decay
        self.chkpt_path = chkpt_path

        self.h1_layer = torch.nn.Linear(*self.input_shape, self.h1_size)
        self.h2_layer = torch.nn.Linear(self.h1_size, self.h2_size)

        # use layer norm b/c it isn't affected by batch size
        # batch norm also fails to copy running avg to target networks
        self.ln1 = torch.nn.LayerNorm(self.h1_size)
        self.ln2 = torch.nn.LayerNorm(self.h2_size)

        # from paper - action vals aren't input until after 2nd hidden layer
        self.action_vals = torch.nn.Linear(self.n_actions, self.h2_size)

        self.out_layer = torch.nn.Linear(self.h2_size, 1)
        self.optimizer = torch.optim.Adam(lr=self.lr, weight_decay=self.decay)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.uniform_(
            self.h1_layer.weight,
            -1 / np.sqrt(*self.input_shape),
            1 / np.sqrt(*self.input_shape),
        )
        torch.nn.init.uniform_(
            self.h2_layer.weight,
            -1 / np.sqrt(self.h1_size),
            1 / np.sqrt(self.h1_size),
        )
        torch.nn.init.uniform_(
            self.out_layer.weight,
            -3e-3,
            3e-3,
        )

    def forward(self, x):
        x = torch.nn.ReLU(self.h1_layer(x))
        x = torch.nn.ReLU(self.h2_layer(x))
        x = torch.nn.BatchNorm1d(x)
        return torch.nn.ReLU(self.out_layer(x))

    def save_checkpoint(self, epoch, loss):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
            },
            self.chkpt_path,
        )

    def load_checkpoint(self, chkpt_path):
        chkpt = torch.load(chkpt_path)
        self.load_state_dict(chkpt["model_state_dict"])
        self.optimizer.load_state_dict(chkpt["optimizer_state_dict"])
        epoch = chkpt["epoch"]
        loss = chkpt["loss"]
        return epoch, loss
