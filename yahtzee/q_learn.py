import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from yahtzee import YahtzeeEnv
from collections import defaultdict


class Agent:
    def __init__(self, gamma=0.99, alpha=0.1):
        self.gamma = gamma
        self.alpha = alpha
        self.Q = defaultdict(lambda: np.zeros(2**5))  # There are 2^5 possible actions.

    def epsilon_greedy_policy(self, state, epsilon):
        state_key = self.state_to_key(state)
        if np.random.rand() < epsilon:
            return np.random.randint(0, 2, size=5)  # Random action
        else:
            return self.greedy_policy(state_key)

    def greedy_policy(self, state_key):
        # Return the action that maximizes the Q-value
        return np.array(
            [int(x) for x in f"{np.argmax(self.Q[state_key]):05b}"]
        )  # Convert index back to binary action

    def update_Q(self, state, action, reward, next_state):
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)

        best_next_action = np.max(self.Q[next_state_key])
        self.Q[state_key][action] += self.alpha * (
            reward + self.gamma * best_next_action - self.Q[state_key][action]
        )

    def state_to_key(self, state):
        # Convert state dictionary to a hashable tuple to use as dictionary keys
        # The potential scores are used directly as part of the state key.
        dice = tuple(state["dice"])
        potential_scores = tuple(state["potential_scores"])
        scored_categories = tuple(state["scored_categories"])
        remaining_rolls = state["remaining_rolls"]
        return (dice, potential_scores, scored_categories, remaining_rolls)


def plot_running_average(scores):
    N = 100
    running_avg = np.empty(len(scores))
    for t in range(len(scores)):
        running_avg[t] = np.mean(scores[max(0, t - N) : (t + 1)])
    plt.plot(running_avg)
    plt.title("Running Average Score Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Average Score")
    plt.show()


if __name__ == "__main__":
    env = YahtzeeEnv()
    agent = Agent()
    n_episodes = 500000
    epsilon = 1.0
    epsilon_step = 1 / n_episodes

    rewards = []

    for episode in range(n_episodes):
        if episode > 0:
            print(
                f"Playing Episode: {episode} of {n_episodes}\t Running Avg: {np.mean(rewards[-100:]):.4f}\t Epsilon: {epsilon:.4f}",
                end="\r",
            )
        state, _ = env.reset()

        terminated, truncated = False, False
        while not terminated and not truncated:
            action = agent.epsilon_greedy_policy(state, epsilon)
            state_, reward, terminated, truncated, _ = env.step(action)
            agent.update_Q(state, action, reward, state_)
            state = state_

        rewards.append(env.get_total_score())
        epsilon = max(epsilon - epsilon_step, 0.01)
    print(" " * 80)

    env.render_scorecard()
    print("Last Game Score:", env.get_total_score())
    plot_running_average(rewards)
