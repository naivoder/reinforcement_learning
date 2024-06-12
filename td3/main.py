import gym
import numpy as np
from agent import DDPGAgent
from utils import plot_running_avg

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == "__main__":
    N_GAMES = 1500

    env = gym.make("BipedalWalker-v3")
    observation_space = env.observation_space.shape
    action_space = env.action_space.shape
    action_bound_low = env.action_space.low
    action_bound_high = env.action_space.high

    agent = DDPGAgent(
        observation_space, action_space, action_bound_low, action_bound_high
    )

    best_score = env.reward_range[0]
    history = list()

    for i in range(N_GAMES):
        state, info = env.reset()

        term, trunc, score = False, False, 0
        while not term and not trunc:
            action = agent.choose_action(state)
            next_state, reward, term, trunc, info = env.step(action)

            agent.store_transition(state, action, reward, next_state, term or trunc)
            agent.learn()

            score += reward
            state = next_state

        history.append(score)
        avg_score = np.mean(history[-100:])

        if score > best_score:
            best_score = score
            agent.save_checkpoints()

        print(
            f"[Episode {i + 1:04}/{N_GAMES}]\tScore = {score:.4f}\tAverage = {avg_score:4f}",
            end="\r",
        )

    plot_running_avg(history)
