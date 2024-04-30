from agent import Agent
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def plot_running_avg(scores):
    avg = np.zeros_like(scores)
    for i in range(len(scores)):
        avg[i] = np.mean(scores[max(0, i - 100) : i + 1])
    plt.plot(avg)
    plt.title("Running Average per 100 Games")
    plt.xlabel("Episode")
    plt.ylabel("Average Score")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    lr = 0.0005
    n_games = 3000

    env = gym.make("LunarLander-v2")
    agent = Agent(lr=lr, input_dims=env.observation_space.shape)

    scores = []
    for i in range(n_games):
        print(f"Playing episode #{i+1}", end="\r")
        state, _ = env.reset()

        score = 0
        terminated, truncated = False, False
        while not terminated and not truncated:
            action = agent.choose_action(state)
            state_, reward, terminated, truncated, _ = env.step(action)
            agent.store_rewards(reward)
            score += reward
            state = state_

        agent.learn()
        scores.append(score)

    plot_running_avg(scores)
