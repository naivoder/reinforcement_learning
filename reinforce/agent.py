from model import LanderNN
import numpy as np
import torch
import torch.nn.functional as F


class Agent:
    """
    Agent class for handling the interaction with the environment, decision making,
    and learning process based on the policy gradient method.

    Parameters
    ----------
    lr : float
        Learning rate for the optimizer.
    input_dims : tuple
        Dimensions of the input state from the environment.
    n_actions : int, optional
        Number of possible actions the agent can take (default is 4).
    gamma : float, optional
        Discount factor for future rewards (default is 0.99).

    Attributes
    ----------
    reward_memory : list
        Memory to store rewards from each action taken.
    action_memory : list
        Memory to store log probabilities of each action taken, for learning.
    policy : LanderNN
        The neural network model that estimates actions from state inputs.

    Methods
    -------
    choose_action(state)
        Determine an action based on current state using the policy network.
    store_rewards(reward)
        Store rewards in the reward memory.
    learn()
        Update the policy network based on stored actions and rewards.
    """

    def __init__(self, lr, input_dims, n_actions=4, gamma=0.99):
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma

        self.reward_memory = []
        self.action_memory = []

        self.policy = LanderNN(
            input_dims=self.input_dims, n_actions=self.n_actions, lr=self.lr
        )

    def choose_action(self, state):
        """
        Determines an action based on the current state using the policy network.

        The state is converted to a tensor, processed through the policy network to get
        action probabilities, which are then used to sample an action.

        Parameters
        ----------
        state : array_like
            The current state from the environment.

        Returns
        -------
        int
            The action to take, represented as an integer.
        """
        # need to add batch dimension to state so pytorch doesn't freak out
        state = torch.Tensor(np.array(state)).to(self.policy.device)
        probs = F.softmax(self.policy(state), dim=-1)

        # build distribution over probabilities
        action_probs = torch.distributions.Categorical(probs)

        # sample action from distribution
        action = action_probs.sample()

        # calculate log prob of action
        log_prob = action_probs.log_prob(action)
        self.action_memory.append(log_prob)

        # dereference pytorch tensor (gym doesn't like torch tensors)
        return action.item()

    def store_rewards(self, reward):
        """
        Stores the given reward in the reward memory.

        Parameters
        ----------
        reward : float
            The reward to store.
        """
        self.reward_memory.append(reward)

    def learn(self):
        """
        Updates the policy network using the action and reward memory.

        This is the learning phase where the gradient descent is performed based
        on the rewards collected and the probabilities of the actions taken.
        """
        self.policy.optimizer.zero_grad()
        Gt = np.zeros_like(self.reward_memory, dtype=np.float64)

        for i in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for j in range(i, len(self.reward_memory)):
                G_sum += discount * self.reward_memory[j]
                discount *= self.gamma
            Gt[i] = G_sum

        Gt = torch.Tensor(Gt).to(self.policy.device)

        loss = 0
        for g, logprob in zip(Gt, self.action_memory):
            loss += -g * logprob
        loss.backward()
        self.policy.optimizer.step()

        self.reward_memory = []
        self.action_memory = []


if __name__ == "__main__":
    agent = Agent(0.0005, (8))
