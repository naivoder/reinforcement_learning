from agent import DiscretePPOAgent
import numpy as np
from utils import plot_running_avg
import gymnasium as gym


env = gym.make("CartPole-v1")

agent = DiscretePPOAgent(
    env.observation_space.shape,
    env.action_space.n,
    alpha=3e-4,
    n_epochs=10,
    batch_size=64,
)

STEPS = 20
N_GAMES = 25000

n_steps, n_learn = 0, 0
scores, best_score = [], env.reward_range[0]
for i in range(N_GAMES):
    state, _ = env.reset()

    term, trunc, score = False, False, 0
    while not term and not trunc:
        action, prob = agent.choose_action(state)

        next_state, reward, term, trunc, _ = env.step(action)
        score += reward

        agent.remember(state, next_state, action, prob, reward, term or trunc)

        n_steps += 1
        if n_steps % STEPS == 0:
            agent.learn()
            n_learn += 1

        state = next_state

    scores.append(score)
    avg_score = np.mean(scores[-100:])

    if avg_score > best_score:
        best_score = avg_score
        # agent.save_checkpoints()

    print(
        f"[Episode {i + 1:04}/{N_GAMES}]\tScore = {score:.4f}\tAverage = {avg_score:4f}",
        end="\r",
    )

plot_running_avg(scores)
