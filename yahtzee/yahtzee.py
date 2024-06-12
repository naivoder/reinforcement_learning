import gymnasium as gym
import numpy as np


class YahtzeeEnv(gym.Env):
    """
    A custom environment for playing Yahtzee.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self):
        super(YahtzeeEnv, self).__init__()
        self.action_space = gym.spaces.MultiDiscrete([2] * 5 + [13])
        self.observation_space = gym.spaces.Dict(
            {
                "dice": gym.spaces.Box(low=1, high=6, shape=(5,), dtype=int),
                "remaining_rolls": gym.spaces.Discrete(3),
                "scorecard": gym.spaces.Box(low=-1, high=50, shape=(13,), dtype=int),
            }
        )

        self.dice = np.random.randint(1, 7, size=(5,))
        self.scorecard = np.full(13, -1, dtype=int)
        self.remaining_rolls = 2
        self.rounds_left = 13

    def reset(self):
        self.dice = np.random.randint(1, 7, size=(5,))
        self.scorecard = np.full(13, -1, dtype=int)
        self.remaining_rolls = 2
        self.rounds_left = 13
        return self.observation(), {}

    def step(self, action):
        reroll_action = action[:5]
        score_action = action[5]

        if self.remaining_rolls > 0 and any(reroll_action):
            self.reroll_dice(reroll_action)
            self.remaining_rolls -= 1
            reward = 0
        else:
            current_score = self.get_total_score()
            self.score_category(score_action)
            reward = self.get_total_score() - current_score
            self.rounds_left -= 1
            self.dice = np.random.randint(1, 7, size=(5,))
            self.remaining_rolls = 2

        return self.observation(), reward, self.rounds_left == 0, False, {}

    def observation(self):
        return {
            "dice": self.dice,
            "remaining_rolls": self.remaining_rolls,
            "scorecard": self.scorecard,
        }

    def reroll_dice(self, reroll_action):
        for i in range(5):
            if reroll_action[i]:
                self.dice[i] = np.random.randint(1, 7)

    def score_category(self, category):
        if self.scorecard[category] == -1:
            score = self.calculate_score(category)
            self.scorecard[category] = score

    def calculate_score(self, category):
        counts = np.bincount(self.dice, minlength=7)[
            1:
        ]  # Count occurrences of each die face
        if category < 6:  # Upper section
            return (category + 1) * counts[category]
        elif category == 6:  # Three of a kind
            return np.sum(self.dice) if np.max(counts) >= 3 else 0
        elif category == 7:  # Four of a kind
            return np.sum(self.dice) if np.max(counts) >= 4 else 0
        elif category == 8:  # Full House
            return 25 if np.any(counts == 3) and np.any(counts == 2) else 0
        elif category == 9:  # Small Straight
            return 30 if self.check_straight(counts, 4) else 0
        elif category == 10:  # Large Straight
            return 40 if self.check_straight(counts, 5) else 0
        elif category == 11:  # Yahtzee
            return 50 if np.max(counts) == 5 else 0
        elif category == 12:  # Chance
            return np.sum(self.dice)

    def check_straight(self, counts, length):
        for i in range(1, 7 - length + 1):
            if all(counts[i : i + length]):
                return True
        return False

    def render(self, mode="human"):
        if mode == "human":
            print("Dice: ", self.dice)
            print("Scorecard: ", self.scorecard)
            print("Remaining rolls: ", self.remaining_rolls)
        elif mode == "ansi":
            return f"Dice: {self.dice} \nScorecard: {self.scorecard} \nRemaining rolls: {self.remaining_rolls}"

    def get_total_score(self):
        adjusted_scores = [max(0, score) for score in self.scorecard]
        upper_section_sum = sum(adjusted_scores[:6])
        total_score = sum(adjusted_scores)
        if upper_section_sum >= 63:
            total_score += 35
        return total_score

    def close(self):
        print("Closing environment and releasing resources...")


if __name__ == "__main__":
    env = YahtzeeEnv()
    obs = env.reset()
    print("Initial observation:", obs)
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        env.render()
        print("Reward:", reward)
    env.close()
