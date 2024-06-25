import numpy as np
from yahtzee import YahtzeeEnv
from ddqn import DuelingDDQNAgent
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
    plt.title(f"Running Average of {metric} Over Last {window} Episodes")
    plt.show()


def train_agent(env, agent, episodes=10000):
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

            next_state, reward, done, _, _ = env.step(action_index)
            next_state = preprocess_state(next_state)
            total_reward += reward

            agent.store_transition(state, action_index, reward, next_state, done)
            state = next_state

        agent.learn()

        total_score = env.get_total_score()
        scores_per_episode.append(total_score)
        rewards_per_episode.append(total_reward)

        avg_score = np.mean(scores_per_episode[-100:])

        if (episode + 1) % 1000 == 0:
            print(
                f"[Episode {episode + 1}/{episodes}] Avg Score: {avg_score}  Epsilon: {agent.epsilon:.4f}"
            )

    return rewards_per_episode, scores_per_episode


if __name__ == "__main__":
    env = YahtzeeEnv()
    state_shape = (5 + 1 + 13,)
    action_space_shape = env.action_space.shape
    num_actions = 44

    agent = DuelingDDQNAgent(
        gamma=0.95,
        epsilon=1.0,
        lr=1e-3,
        action_space_shape=action_space_shape,
        input_dims=state_shape,
        batch_size=64,
        eps_min=0.1,
        eps_dec=1e-6,
        replace=1000,
        chkpt_dir="weights/ddqn",
    )

    rewards, scores = train_agent(env, agent, episodes=100000000)
    env.close()

    plot_running_average(rewards, window=1000, metric="Rewards")
    plot_running_average(scores, window=1000, metric="Scores")
