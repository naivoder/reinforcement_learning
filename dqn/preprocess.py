import gymnasium as gym
import numpy as np
from collections import deque
import cv2


class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env, repeat=4):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.env = env
        self.repeat = repeat
        self.frame_buffer = np.zeros(
            (2, *self.env.observation_space.shape), dtype=np.float32
        )

    def step(self, action):
        total_reward = 0
        term, trunc = False, False

        for i in range(self.repeat):
            state, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            self.frame_buffer[i] = state

            if term or trunc:
                break

        # max_frame = np.max(self.frame_buffer, axis=0)
        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, total_reward, term, trunc, info

    def reset(self, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)
        self.frame_buffer = np.zeros(
            (2, *self.env.observation_space.shape), dtype=np.float32
        )
        self.frame_buffer[0] = state

        return state, info


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super(PreprocessFrame, self).__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(0.0, 1.0, self.shape, dtype=np.float32)

    def observation(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, self.shape, interpolation=cv2.INTER_AREA)
        return state / 255.0


class StackFrames(gym.Wrapper):
    def __init__(self, env, size=4):
        super(StackFrames, self).__init__(env)
        self.size = int(size)
        self.stack = deque([], maxlen=self.size)

        shape = self.env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, (self.size, *shape), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)
        self.stack = deque([state] * self.size, maxlen=self.size)
        return np.array(self.stack), info

    def step(self, action):
        state, reward, term, trunc, info = self.env.step(action)
        self.stack.append(state)
        return np.array(self.stack), reward, term, trunc, info


class AtariEnv:
    def __init__(self, env, shape=(84, 84), max_frame=2, size=4):
        self.env = gym.make(env)
        self.env = RepeatActionAndMaxFrame(self.env, max_frame)
        self.env = PreprocessFrame(self.env, shape)
        self.env = StackFrames(self.env, size)

    def make(self):
        return self.env


if __name__ == "__main__":
    env = AtariEnv("ALE/Pong-v5").make()
    state, _ = env.reset()

    print("Expected Shape:", env.observation_space.shape)
    print("Actual Shape:", state.shape)
