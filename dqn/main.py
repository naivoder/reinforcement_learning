import numpy as np
from dqn import DQNAgent
from preprocess import AtariEnv
import utils
import os
import warnings

warnings.simplefilter("ignore")
os.makedirs("weights", exist_ok=True)

if __name__ == "__main__":
    N_GAMES = 500

    env_name = "ALE/Pong-v5"
    save_name = env_name.split("/")[-1]
    env = AtariEnv(env_name).make()

    agent = DQNAgent(
        env_name,
        input_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        mem_size=1e5,
        batch_size=32,
        eps_dec=1e-5,
        replace_target_count=1000,
    )

    best_score = env.reward_range[0]
    scores = []
    for i in range(N_GAMES):
        state, _ = env.reset()

        score = 0
        term, trunc = False, False
        while not term and not trunc:
            action = agent.choose_action(state)

            next_state, reward, term, trunc, _ = env.step(action)
            score += reward

            agent.store_transition(state, action, reward, next_state, term or trunc)
            agent.learn()

            state = next_state

        scores.append(score)
        avg_score = np.mean(scores[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_checkpoint()

        print(
            f"[Episode {i + 1:04}/{N_GAMES}]\tScore = {score:.4f}\tAverage = {avg_score:4f}",
            end="\r",
        )

    utils.plot_running_avg(scores)
