import gymnasium as gym
import utils
from agent import SACAgent
import numpy as np
from collections import UserDict
import gym.envs.registration

# Do this before importing pybullet_envs
# (adds an extra property env_specs as a property to the registry,
# so it looks like the <0.26 envspec version)
registry = UserDict(gym.envs.registration.registry)
registry.env_specs = gym.envs.registration.registry
gym.envs.registration.registry = registry

import pybullet_envs

N_GAMES = 250

env = gym.make("InvertedPendulumBulletEnv-v0")
agent = SACAgent(
    env.observation_space.shape,
    env.action_space,
    tau=5e-3,
    reward_scale=2,
    batch_size=256,
)

scores = []
for i in range(N_GAMES):
    state = env.reset()

    term, trunc, score = False, False, 0
    while not term and not trunc:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        score += reward

        agent.store_transition(state, action, reward, next_state, done)
        agent.learn()

        state = next_state

    scores.append(score)
    avg_score = np.mean(scores[-100:])

    if avg_score > best_score:
        best_score = avg_score
        agent.save_checkpoints()

    print(
        f"[Episode {i + 1:04}/{N_GAMES}]\tScore = {score:.4f}\tAverage = {avg_score:4f}",
        end="\r",
    )

utils.plot_running_avg(scores)
