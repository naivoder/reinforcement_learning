import gym
import numpy as np
from agent import DDPGAgent
from utils import plot_running_avg

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == "__main__":
    N_GAMES = 1000

    env = gym.make("LunarLanderContinuous-v2")
    agent = DDPGAgent(env.observation_space.shape, env.action_space.shape)

    best_score = env.reward_range[0]
    history = list()

    for i in range(N_GAMES):
        state, info = env.reset()
        agent.action_noise.reset()

        term, trunc, score = False, False, 0
        while not term and not trunc:
            action = agent.choose_action(state)
            next_state, reward, term, trunc, info = env.step(action)
            # done = True if term or trunc else False

            agent.store_transition(state, action, reward, next_state, term or trunc)
            agent.learn()

            score += reward
            state = next_state

        history.append(score)
        avg_score = np.mean(history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_checkpoints(i + 1, score)

        print(
            f"[Episode {i + 1:04}/{N_GAMES}]\tScore = {score:.4f}\tAverage = {avg_score:4f}",
            end="\r",
        )

    plot_running_avg(history)
