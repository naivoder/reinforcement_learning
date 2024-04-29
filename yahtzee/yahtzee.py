import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class YahtzeeEnv(gym.Env):
    """
    A custom environment for playing Yahtzee, compatible with the OpenAI Gym framework.

    Attributes
    ----------
    action_space : gym.spaces
        Contains:
        - 'dice_action': MultiBinary(5), where each bit decides if a corresponding dice should be rerolled.
    observation_space : gym.spaces
        Contains:
        - 'dice': Box, the current dice values (1-6).
        - 'scorecard': Box, current score for each category (-1 if unscored, 0-50 if scored).
        - 'remaining_rolls': Discrete, number of remaining rolls in the current round (0-2).
    dice : ndarray
        Current values of the dice.
    scorecard : ndarray
        Current scores or -1 if a category has not been scored yet.
    rounds_left : int
        Number of rounds left in the game.
    rolls_this_round : int
        Number of rolls taken in the current round.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self):
        super(YahtzeeEnv, self).__init__()
        self.action_space = gym.spaces.MultiBinary(5)

        self.observation_space = gym.spaces.Dict(
            {
                "dice": gym.spaces.Box(low=1, high=6, shape=(5,), dtype=int),
                "scored_categories": gym.spaces.MultiBinary(13),
                "remaining_rolls": gym.spaces.Discrete(3),
            }
        )

        self.dice = None
        self.scored_categories = None
        self.actual_scores = None
        self.potential_scores = None
        self.rounds_left = 13
        self.rolls_this_round = 0

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns
        -------
        tuple
            Initial observation of the environment and an empty info dictionary.
        """
        self.dice = np.random.randint(1, 7, size=(5,))
        self.scored_categories = np.zeros(13, dtype=int)
        self.actual_scores = np.full(13, -1, dtype=int)
        self.potential_scores = np.zeros(13, dtype=int)
        self.rounds_left = 13
        self.rolls_this_round = 0
        self.update_potential_scores()
        return self.observation(), {}

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        if self.rolls_this_round < 2 and any(action):
            self.reroll_dice(action)
            self.rolls_this_round += 1
            self.update_potential_scores()
            return self.observation(), 0, False, False, {}

        score_action = self.select_highest_scoring_category()
        reward = self.score_category(score_action)
        self.rounds_left -= 1
        terminated = self.rounds_left == 0
        self.dice = np.random.randint(1, 7, size=(5,))
        self.rolls_this_round = 0
        self.update_potential_scores()
        return self.observation(), reward, terminated, False, {}

    def observation(self):
        return {
            "dice": self.dice,
            "scored_categories": self.scored_categories,
            "remaining_rolls": 3 - self.rolls_this_round,
        }

    def reroll_dice(self, dice_action):
        for i in range(5):
            if dice_action[i]:
                self.dice[i] = np.random.randint(1, 7)

    def score_category(self, category):
        if self.scored_categories[category] == 0:
            score = self.potential_scores[category]
            self.actual_scores[category] = score
            self.scored_categories[category] = 1
            return score
        return -10

    def select_highest_scoring_category(self):
        valid_scores = [
            score if not scored else -1
            for score, scored in zip(self.potential_scores, self.scored_categories)
        ]
        return np.argmax(valid_scores)

    def update_potential_scores(self):
        counts = np.bincount(self.dice, minlength=7)[
            1:
        ]  # Count occurrences of each die face
        for i in range(6):  # Upper section scoring
            if self.scored_categories[i] == 0:
                self.potential_scores[i] = (i + 1) * counts[i]
        # Scoring for combinations
        self.potential_scores[6] = (
            np.sum(self.dice) if np.max(counts) >= 3 else 0
        )  # Three of a kind
        self.potential_scores[7] = (
            np.sum(self.dice) if np.max(counts) >= 4 else 0
        )  # Four of a kind
        self.potential_scores[8] = (
            25 if np.any(counts == 3) and np.any(counts == 2) else 0
        )  # Full House
        self.potential_scores[9] = (
            30 if self.check_straight(counts, 4) else 0
        )  # Small Straight
        self.potential_scores[10] = (
            40 if self.check_straight(counts, 5) else 0
        )  # Large Straight
        self.potential_scores[11] = 50 if np.max(counts) == 5 else 0  # Yahtzee
        self.potential_scores[12] = np.sum(self.dice)  # Chance

    def check_straight(self, counts, length):
        for i in range(1, 7 - length + 1):
            if all(counts[i : i + length]):
                return True
        return False

    def render(self, mode="human"):
        """
        Render the observation space.

        Parameters
        ----------
        mode : str
            Render mode to use ('human' for console output, 'ansi' for string).

        Returns
        -------
        None or str
            None if 'human', string if 'ansi'.
        """
        if mode == "human":
            print("Dice: ", self.dice)
            print("Scorecard: ", self.scorecard)
            print("Rolls left this round: ", 3 - self.rolls_this_round)
        elif mode == "ansi":
            return f"Dice: {self.dice} \nScorecard: {self.scorecard} \nRolls left this round: {3 - self.rolls_this_round}"

    def render_scorecard(self):
        """Renders the scorecard in a tabular format to the console."""
        titles = [
            "Ones",
            "Twos",
            "Threes",
            "Fours",
            "Fives",
            "Sixes",
            "Three of a Kind",
            "Four of a Kind",
            "Full House",
            "Small Straight",
            "Large Straight",
            "Yahtzee",
            "Chance",
        ]
        print("+-------------------+-------+")
        print("| Category          | Score |")
        print("+-------------------+-------+")
        for i, title in enumerate(titles):
            score_display = (
                "-" if self.actual_scores[i] == -1 else str(self.actual_scores[i])
            )
            print(f"| {title:<17} | {score_display:<5} |")
        print("+-------------------+-------+")

    def get_total_score(self):
        """
        Calculates and returns the total score.

        Returns
        -------
        int
            The total score calculated from the scorecard.
        """
        total_score = sum(self.actual_scores)
        return total_score

    def close(self):
        """Releases any resources held by the environment."""
        print("Closing environment and releasing resources...")


def plot_scores(scores):
    """
    Plots a list of scores over time using matplotlib.

    Parameters
    ----------
    scores : list of int
        The scores to plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(scores, marker="o", linestyle="-", color="b")
    plt.title("Yahtzee Scores Over Time")
    plt.xlabel("Game Number")
    plt.ylabel("Score")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    env = YahtzeeEnv()
    debug = False

    scores = []
    n_games = 3

    for i in range(n_games):
        print(f"\nPlaying Game {i+1} of {n_games}")

        obs = env.reset()
        terminated, truncated = False, False

        while not terminated and not truncated:
            action = env.action_space.sample()  # Randomly decide which dice to re-roll
            obs, reward, terminated, truncated, info = env.step(action)
            if debug:
                env.render()

        env.render_scorecard()
        scores.append(env.get_total_score())

    plot_scores(scores)
    env.close()
