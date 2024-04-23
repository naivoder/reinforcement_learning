import numpy as np
import gymnasium as gym


class Agent:
    """
    A simple agent for the CartPole environment that uses Temporal Difference (TD) learning
    to estimate state values based on the angle of the pole.

    Attributes
    ----------
    bins : numpy.ndarray
        Bins for discretizing the pole angle into discrete states.
    action_space : list
        List of possible actions. For CartPole, it is [0, 1] representing left and right movements.
    V : dict
        A dictionary mapping discretized pole angles to their estimated values.
    gamma : float
        Discount factor, dictates the importance of future rewards.
    alpha : float
        Learning rate, determines the step size at each iteration.

    Methods
    -------
    init_V()
        Initializes the value estimates for all discretized pole angles.
    policy(state)
        Determines the action to take based on the current state.
    update_V(state, reward, next_state)
        Updates the value estimate for the given state.
    """

    def __init__(self, gamma=0.99, alpha=0.1):
        """
        Parameters
        ----------
        gamma : float, optional
            The discount factor for future rewards (default is 0.99).
        alpha : float, optional
            The learning rate or step size (default is 0.1).
        """
        self.bins = np.linspace(-0.2095, 0.2095, 10)
        self.action_space = [0, 1]
        self.V = {}
        self.gamma = gamma
        self.alpha = alpha

        self.init_V()

    def init_V(self):
        """Initializes value estimates for all discretized pole angles to zero."""
        for angle in range(len(self.bins) + 1):
            self.V[angle] = 0

    def policy(self, state):
        """
        Determines the action based on the current state, specifically the angle of the pole.

        Parameters
        ----------
        state : numpy.ndarray
            The current state of the environment, including the pole angle.

        Returns
        -------
        int
            The action to be taken: 0 for left, 1 for right.
        """
        return self.action_space[0] if state[2] < 0 else self.action_space[1]

    def update_V(self, state, reward, next_state):
        """
        Updates the value estimate for the current state using the reward and the value of the next state.

        Parameters
        ----------
        state : numpy.ndarray
            The current state of the environment.
        reward : float
            The reward received from the environment after taking the action.
        next_state : numpy.ndarray
            The state of the environment after taking the action.

        Returns
        -------
        None
        """
        angle = np.digitize(state[2], self.bins)
        next_angle = np.digitize(next_state[2], self.bins)

        self.V[angle] = self.V[angle] + self.alpha * (
            reward + self.gamma * self.V[next_angle] - self.V[angle]
        )


def main():
    """
    Main function to execute the CartPole TD-Learning Agent.

    Creates an instance of the environment and the agent, then performs episodes of
    training with the agent making decisions based on its policy and updating its value estimates.
    """
    env = gym.make("CartPole-v1")

    agent = Agent()
    n_episodes = 50000

    for episode in range(n_episodes):
        print("Executing episode:", episode + 1, end="\r")
        state, _ = env.reset()

        terminated, truncated = False, False
        while not terminated and not truncated:
            action = agent.policy(state)
            state_, reward, terminated, truncated, _ = env.step(action)
            agent.update_V(state, reward, state_)
            state = state_

    print("Training Complete!\t\t")
    print("V:", list(agent.V.values()))


if __name__ == "__main__":
    main()
