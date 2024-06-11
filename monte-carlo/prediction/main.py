import numpy as np
import gymnasium as gym


class Agent:
    """
    A simple agent that uses a Monte Carlo learning algorithm to estimate the value of states in a
    blackjack game environment provided by Gym.

    Attributes
    ----------
    V : dict
        A dictionary to store the estimated values for each state.
    sum_space : list
        A list of possible sums the player can have (from 4 to 21).
    dealer_show_card_space : list
        A list of possible cards the dealer can show (from 1 to 10).
    ace_space : list
        A list indicating whether the player holds a usable ace (True or False).
    action_space : list
        List of possible actions, where 0 is 'stick' and 1 is 'hit'.
    state_space : list
        List of all possible states derived from sum_space, dealer_show_card_space, and ace_space.
    returns : dict
        A dictionary to store the returns for each state.
    states_visited : dict
        Tracks whether each state has been visited in an episode to implement first-visit MC.
    memory : list
        Temporary storage of state and reward history in an episode.
    gamma : float
        Discount factor which determines the importance of future rewards.

    Methods
    -------
    init_vals()
        Initializes the state values, returns, and visits for all possible states.
    policy(state)
        Defines the policy to be used by the agent to decide actions based on the current state.
    update_V()
        Updates the state value estimates based on accumulated episode returns.
    """

    def __init__(self, gamma=0.99):
        """
        Parameters
        ----------
        gamma : float, optional
            The discount factor for future rewards (default is 0.99).
        """
        self.V = {}
        self.sum_space = [i for i in range(4, 22)]
        self.dealer_show_card_space = [i for i in range(1, 11)]
        self.ace_space = [False, True]
        self.action_space = [0, 1]

        self.state_space = []
        self.returns = {}
        self.states_visited = {}
        self.memory = []
        self.gamma = gamma

        self.init_vals()

    def init_vals(self):
        """Initializes state values, returns, and visitation counts for all possible states."""
        for total in self.sum_space:
            for card in self.dealer_show_card_space:
                for ace in self.ace_space:
                    self.V[(total, card, ace)] = 0
                    self.returns[(total, card, ace)] = []
                    self.states_visited[(total, card, ace)] = 0
                    self.state_space.append((total, card, ace))

    def policy(self, state):
        """
        Determines the action to take in a given state. The policy is simple: stick if the total is 20 or 21,
        otherwise hit.

        Parameters
        ----------
        state : tuple
            The current state represented as (total, dealer's card, usable ace).

        Returns
        -------
        int
            The action to be taken, where 0 means 'stick' and 1 means 'hit'.
        """
        total, _, _ = state
        return 0 if total >= 20 else 1

    def update_V(self):
        """
        Updates the value estimates for states visited in an episode. This is the core of the Monte Carlo
        learning method, where the returns following the first visits to states are averaged to estimate values.
        """
        for idt, (state, _) in enumerate(self.memory):
            G = 0
            if self.states_visited[state] == 0:
                self.states_visited[state] += 1
                discount = 1  # gamma^0 == 1
                for t, (_, reward) in enumerate(self.memory[idt:]):
                    G += reward * discount
                    discount *= self.gamma
                    self.returns[state].append(G)

        for state, _ in self.memory:
            self.V[state] = np.mean(self.returns[state])

        for state in self.state_space:
            self.states_visited[state] = 0

        self.memory = []


if __name__ == "__main__":
    env = gym.make("Blackjack-v1")
    agent = Agent()
    n_episodes = 500000

    for i in range(n_episodes):
        print("Executing episode:", i, end="\r")

        observation, info = env.reset()
        terminated, truncated = False, False

        while not terminated or truncated:
            action = agent.policy(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            print(info)
            agent.memory.append((observation, reward))
            observation = observation_

        agent.update_V()

    print("Value of Likely Win State:", agent.V[21, 3, True])
    print("Value of Likely Lose State:", agent.V[4, 1, False])
