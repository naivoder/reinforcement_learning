import numpy as np
from ddqn import DDQNAgent
from preprocess import AtariEnv
import utils
import os
import warnings

warnings.simplefilter("ignore")
os.makedirs("weights", exist_ok=True)

if __name__ == "__main__":
    N_GAMES = 500

    env_name = "ALE/BankHeist-v5"
    save_name = env_name.split("/")[-1]
    env = AtariEnv(env_name).make()

    print(f"\nEnvironment: {save_name}")
    print(f"Obs.Space: {env.observation_space.shape} Act.Space: {env.action_space.n}")

    agent = DDQNAgent(
        save_name,
        input_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        mem_size=10000,
        batch_size=32,
        eps_dec=1e-5,
        replace_target_count=1000,
    )

    best_score = env.reward_range[0]
    scores = []
    n_steps = 0
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

            n_steps += 1
            state = next_state

        scores.append(score)
        avg_score = np.mean(scores[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_checkpoint()

        game_str = f"[Game {i + 1:04}/{N_GAMES}]"
        score_str = f"   Score = {score:.2f}"
        avg_str = f"   Average = {avg_score:.2f}"
        step_str = f"   Steps = {n_steps}"
        eps_str = f"   Epsilon = {agent.epsilon:.4f}"
        print(game_str + score_str + avg_str + step_str + eps_str, end="\r")

    utils.plot_running_avg(scores)
