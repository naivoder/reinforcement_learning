import gymnasium as gym
import numpy as np
from ddqn import DDQNAgent
from preprocess import AtariEnv
import utils
import os
import warnings

warnings.simplefilter("ignore")
os.makedirs("weights", exist_ok=True)


def make_env():
    return AtariEnv(env_name, clip_rewards=True).make()


if __name__ == "__main__":
    N_STEPS = 10000
    N_ENVS = 32

    env_name = "PongNoFrameskip-v4"
    save_name = env_name.split("/")[-1]

    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(N_ENVS)])

    print(f"\nEnvironment: {save_name}")
    print(
        f"Obs.Space: {envs.single_observation_space.shape} Act.Space: {envs.single_action_space.n}"
    )

    agent = DDQNAgent(
        save_name,
        input_shape=envs.single_observation_space.shape,
        n_actions=envs.single_action_space.n,
        mem_size=100000,
        batch_size=32,
        eps_dec=1e-5,
        replace_target_count=1000,
    )

    best_score = -np.inf
    avg_score = 0.00
    scores = []
    score = np.zeros(N_ENVS)

    states, _ = envs.reset()
    for i in range(N_STEPS):
        actions = [agent.choose_action(state) for state in states]

        next_states, rewards, term, trunc, _ = envs.step(actions)

        for j in range(N_ENVS):
            agent.store_transition(
                states[j],
                actions[j],
                rewards[j],
                next_states[j],
                term[j] or trunc[j],
            )
            agent.learn()

            score[j] += rewards[j]
            if term[j] or trunc[j]:
                scores.append(score[j])
                score[j] = 0

        states = next_states

        if len(scores) > 0:
            avg_score = np.mean(scores[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_checkpoint()

        ep_str = f"[Epoch {i + 1:04}/{N_STEPS}]"
        score_str = f"  Score = {np.mean(score):.2f}"
        g_str = f"  Games = {len(scores)}"
        avg_str = f"  Average = {avg_score:.2f}"
        eps_str = f"  Epsilon = {agent.epsilon:.4f}"
        sts_str = f"  Steps = {i * N_ENVS}"
        print(ep_str + g_str + score_str + avg_str + eps_str + sts_str, end="\r")

    utils.plot_running_avg(scores)
