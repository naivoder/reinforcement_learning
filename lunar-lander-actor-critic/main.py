import torch
import numpy as np
import gymnasium as gym
from torch.nn.functional import softmax
from utils import plot_running_avg


class PolicyNetwork(torch.nn.Module):
    """
    A neural network for implementing a policy gradient based reinforcement learning algorithm.

    This network includes an input layer, one hidden layer, and two output layers (actor and critic).
    The actor outputs a probability distribution over actions, and the critic outputs a scalar value estimate.

    Parameters
    ----------
    input_shape : tuple
        The shape of the input data.
    n_actions : int
        The number of possible actions in the environment.
    learning_rate : float, optional
        The learning rate for the optimizer, by default 5e-6.

    Attributes
    ----------
    input : torch.nn.Linear
        The input layer that linearly transforms the input data.
    activation : torch.nn.ReLU
        The ReLU activation function applied after linear transformations.
    hidden : torch.nn.Linear
        The hidden layer that linearly transforms data from the input layer.
    actor_out : torch.nn.Linear
        The output layer for the actor part, producing a vector of action scores.
    critic_out : torch.nn.Linear
        The output layer for the critic part, producing a value estimate.
    optimizer : torch.optim.Adam
        The optimizer used to update network weights.
    """

    def __init__(self, input_shape, n_actions, learning_rate=5e-6):
        super(PolicyNetwork, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.lr = learning_rate
        self.input = torch.nn.Linear(*self.input_shape, 2000)
        self.activation = torch.nn.ReLU()
        self.hidden = torch.nn.Linear(2000, 1500)
        self.actor_out = torch.nn.Linear(1500, self.n_actions)
        self.critic_out = torch.nn.Linear(1500, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor containing the state.

        Returns
        -------
        tuple
            A tuple containing:
            - action (torch.Tensor): The action probabilities from the actor output.
            - value (torch.Tensor): The value estimate from the critic output.
        """
        x = self.input(x)
        x = self.activation(x)
        x = self.hidden(x)
        x = self.activation(x)
        action = self.actor_out(x)
        value = self.critic_out(x)
        return action, value


class ActorCritic:
    """
    An implementation of the Actor-Critic algorithm using the PolicyNetwork.

    Parameters
    ----------
    input_shape : tuple
        The shape of the input to the PolicyNetwork.
    n_actions : int
        The number of actions available in the environment.
    gamma : float, optional
        The discount factor for future rewards, default is 0.99.

    Attributes
    ----------
    device : str
        The device (CPU or CUDA) on which the network will be run.
    policy_net : PolicyNetwork
        The policy network which provides actor and critic functionalities.
    gamma : float
        The discount factor for calculating future rewards.
    log_prob : torch.Tensor or None
        The log probability of the most recent action taken.
    """

    def __init__(self, input_shape, n_actions, gamma=0.99):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy_net = PolicyNetwork(input_shape, n_actions).to(self.device)
        self.gamma = gamma
        self.log_prob = None  # log prob of most recent action

    def choose_action(self, state):
        """
        Choose an action based on the current state using the policy network.

        Parameters
        ----------
        state : array-like
            The current state of the environment.

        Returns
        -------
        int
            The action chosen by the actor component of the policy network.
        """
        state = torch.Tensor(np.array(state)).to(self.device)
        actions, _ = self.policy_net(state)

        action_probs = torch.distributions.Categorical(softmax(actions, dim=-1))
        action = action_probs.sample()
        self.log_prob = action_probs.log_prob(action)

        return action.item()

    def learn(self, state, reward, next_state, done):
        """
        Update the weights of the policy network based on the reward received and the transition.

        This function computes the loss for both the actor and critic components and updates the network weights.

        Parameters
        ----------
        state : array-like
            The state from which the action was taken.
        reward : float
            The reward received after taking the action.
        next_state : array-like
            The state after taking the action.
        done : bool
            A flag indicating whether the episode has ended.

        Returns
        -------
        None
        """
        self.policy_net.optimizer.zero_grad()

        state = torch.Tensor(np.array(state)).to(self.device)
        next_state = torch.Tensor(np.array(next_state)).to(self.device)
        reward = torch.Tensor([reward]).to(self.device)

        _, this_value = self.policy_net(state)
        _, next_value = self.policy_net(next_state)

        delta = reward + self.gamma * (next_value * (1 - int(done))) - this_value
        actor_loss = -self.log_prob * delta
        critic_loss = delta**2
        loss = actor_loss + critic_loss

        loss.backward()
        self.policy_net.optimizer.step()


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    agent = ActorCritic(env.observation_space.shape, 4)
    n_episodes = 2000

    avg_score = 0
    scores = []
    for i in range(n_episodes):
        print(f" Playing episode: {i+1}\t Avg Score: {avg_score:.4f}", end="\r")
        state, info = env.reset()
        score = 0

        terminated, truncated = False, False
        while not terminated and not truncated:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = True if terminated or truncated else False
            agent.learn(state, reward, next_state, done)
            score += reward

        scores.append(score)
        print(" " * 64, end="\r")
        if i % 10 == 0:
            avg_score = np.mean(scores[-10:])

    plot_running_avg(scores)
