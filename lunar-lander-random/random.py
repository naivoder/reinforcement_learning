import gymnasium as gym

if __name__ == "__main__":
    env = gym.make("LunarLander-v2", render_mode="human")
    n_games = 10

    for i in range(n_games):
        print(f"\nPlaying Game #{i+1}")
        state, _ = env.reset()

        score = 0
        terminated, truncated = False, False
        while not terminated and not truncated:
            action = env.action_space.sample()
            state_, reward, terminated, truncated, _ = env.step(action)
            score += reward
            state = state_
            env.render()

        print("Final Score:", score)
    env.close()
