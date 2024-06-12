import numpy as np
from yahtzee import YahtzeeEnv
from agent import DuelingDDQNAgent
import matplotlib.pyplot as plt


def preprocess_state(state):
    state = np.concatenate(
        [state["dice"], [state["remaining_rolls"]], state["scorecard"]]
    )
    return state


def plot_running_average(rewards, window=100, metric="Rewards"):
    running_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
    plt.plot(running_avg)
    plt.xlabel("Episode")
    plt.ylabel(f"Running Average of Last {window} {metric}")
    plt.title(f"Running Average of Rewards Over Last {window} Episodes")
    plt.show()


def train_agent(env, agent, episodes=1000):
    rewards_per_episode, scores_per_episode = [], []

    for episode in range(episodes):
        state, _ = env.reset()
        state = preprocess_state(state)
        total_reward = 0
        done = False

        while not done:
            valid_categories = [
                i for i, score in enumerate(state[-13:]) if score == -1
            ]  # Unscored categories
            action_index = agent.choose_action(state, valid_categories)

            action = np.unravel_index(action_index, env.action_space.nvec)
            next_state, reward, done, _, _ = env.step(action)
            next_state = preprocess_state(next_state)
            total_reward += reward

            agent.store_transition(state, action_index, reward, next_state, done)
            agent.learn()
            state = next_state

        total_score = env.get_total_score()
        scores_per_episode.append(total_score)
        rewards_per_episode.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(
                f"[Episode {episode + 1}/{episodes}] Score: {total_score}  Epsilon: {agent.epsilon:.4f}"
            )

    return rewards_per_episode, scores_per_episode


if __name__ == "__main__":
    env = YahtzeeEnv()
    state_shape = (5 + 1 + 13,)
    action_space_shape = env.action_space.nvec
    num_actions = np.prod(env.action_space.nvec)

    agent = DuelingDDQNAgent(
        gamma=0.99,
        epsilon=1.0,
        lr=0.0001,
        action_space_shape=action_space_shape,
        input_dims=state_shape,
        batch_size=128,
        eps_min=0.1,
        eps_dec=1e-5,
        replace=100,
        chkpt_dir="weights/ddqn",
    )

    rewards, scores = train_agent(env, agent, episodes=10000)
    env.close()

    plot_running_average(rewards, window=100, metric="Rewards")
    plot_running_average(scores, window=100, metric="Scores")
