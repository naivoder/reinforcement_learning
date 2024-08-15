import gymnasium as gym
import numpy as np
from agent import DuelingDDQNAgent
from preprocess import AtariEnv
import utils
import os
import warnings

warnings.simplefilter("ignore")
os.makedirs("weights", exist_ok=True)


def make_env():
    return AtariEnv(env_name, clip_rewards=True).make()


if __name__ == "__main__":
    N_STEPS = 300

    env_name = "PongNoFrameskip-v4"
    save_name = env_name.split("/")[-1]

    env = make_env()

    print(f"\nEnvironment: {save_name}")
    print(f"Obs.Space: {env.observation_space.shape} Act.Space: {env.action_space.n}")

    agent = DuelingDDQNAgent(
        save_name,
        input_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        mem_size=50000,
        batch_size=32,
        eps_dec=1e-5,
        replace_target_count=1000,
    )

    best_score = -np.inf
    avg_score = 0.00
    scores = []
    score = 0

    state, _ = env.reset()
    while len(scores) < N_STEPS:
        action = int(agent.choose_action(state))
        next_state, reward, term, trunc, _ = env.step(action)

        agent.store_transition(state, action, reward, next_state, term or trunc)

        score += reward
        if term or trunc:
            scores.append(score)
            score = 0
            state, _ = env.reset()
        else:
            state = next_state

        agent.learn()

        if len(scores) > 0:
            avg_score = np.mean(scores[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_checkpoint()

        g_str = f"Games = {len(scores)}"
        avg_str = f"  Average = {avg_score:.2f}"
        eps_str = f"  Epsilon = {agent.epsilon:.4f}"
        print(g_str + avg_str + eps_str, end="\r")

    utils.plot_running_avg(scores)
