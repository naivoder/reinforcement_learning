import gymnasium as gym
import utils
from agent import SACAgent
import numpy as np
import torch 
import os 

os.makedirs('weights', exist_ok=True)

print("Training on:", "GPU" if torch.cuda.is_available() else "CPU")

N_GAMES = 250

env = gym.make("Pusher-v4")

agent = SACAgent(
    env.observation_space.shape,
    env.action_space,
    tau=5e-3,
    reward_scale=2,
    batch_size=64,
)

scores, best_score = [], 0
for i in range(N_GAMES):
    state, _ = env.reset()

    term, trunc, score = False, False, 0
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
        agent.save_checkpoints()

    print(
        f"[Episode {i + 1:04}/{N_GAMES}]\tScore = {score:.4f}\tAverage = {avg_score:4f}",
        end="\r",
    )

utils.plot_running_avg(scores)
