from agent import Agent
import gymnasium as gym
import matplotlib.pyplot as plt


def plot_running_avg(scores):
    avg = []
    for i in range(len(scores)):
        n = 100 if i >= 100 else i
        avg.append(sum(scores[i - n : i + 1]) / n)
    plt.plot(avg)
    plt.title("Running Average per 100 Games")


if __name__ == "__main__":
    lr = 0.0005
    n_games = 3000

    env = gym.make("LunarLander-v2", render_mode="human")
    agent = Agent(lr=lr, input_dims=env.observation_space.shape)

    scores = []
    for i in range(len(n_games)):
        print(f"Playing episode #{i+1}", end="\r")
        state, _ = env.reset()

        score = 0
        terminated, truncated = False, False
        while not terminated and not truncated:
            action = agent.choose_action(state)
            state_, reward, terminated, trucated, _ = env.step(action)
            agent.store_rewards(reward)
            score += reward

        agent.learn()
        scores.append(score)

    plot_running_avg(scores)
