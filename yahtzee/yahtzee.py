import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class YahtzeeEnv(gym.Env):
    """
    A custom environment for playing Yahtzee, compatible with the OpenAI Gym framework.

    Attributes
    ----------
    action_space : gym.spaces
        A dictionary space containing:
        - 'dice_action': MultiBinary(5), where each bit decides if a corresponding dice should be rerolled.
        - 'score_action': Discrete(13), selecting which score category to use (0-12).
    observation_space : gym.spaces
        A dictionary space containing:
        - 'dice': Box, the current dice values (1-6).
        - 'scorecard': Box, current score for each category (-1 if unscored, 0-50 if scored).
        - 'potential_scores': Box, potential scores that can be obtained in each category.
        - 'remaining_rolls': Discrete, number of remaining rolls in the current round (0-2).
    dice : ndarray
        Current values of the dice.
    scorecard : ndarray
        Current scores or -1 if a category has not been scored yet.
    potential_scores : ndarray
        Potential scores for each category based on the current dice.
    rounds_left : int
        Number of rounds left in the game.
    rolls_this_round : int
        Number of rolls taken in the current round.

    Methods
    -------
    reset(seed=None, options=None)
        Resets the environment to the initial state and returns the initial observation.
    step(action)
        Executes a given action and returns the new state, reward, done, and info.
    render(mode='human')
        Renders the current state of the environment.
    close()
        Performs any necessary cleanup.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self):
        """Initializes a new Yahtzee game environment."""
        super(YahtzeeEnv, self).__init__()
        self.action_space = gym.spaces.Dict(
            {
                "dice_action": gym.spaces.MultiBinary(5),
                "score_action": gym.spaces.Discrete(13),
            }
        )

        self.observation_space = gym.spaces.Dict(
            {
                "dice": gym.spaces.Box(low=1, high=6, shape=(5,), dtype=int),
                "scorecard": gym.spaces.Box(low=-1, high=50, shape=(13,), dtype=int),
                "potential_scores": gym.spaces.Box(
                    low=0, high=50, shape=(13,), dtype=int
                ),
                "remaining_rolls": gym.spaces.Discrete(3),
            }
        )

        self.dice = None
        self.scorecard = None
        self.potential_scores = None
        self.rounds_left = 13
        self.rolls_this_round = 0

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator.
        options : dict, optional
            Additional options for resetting the environment (unused).

        Returns
        -------
        tuple
            Initial observation of the environment and an empty info dictionary.
        """
        super().reset(seed=seed)
        self.dice = np.random.randint(1, 7, size=(5,))
        self.scorecard = np.full(13, -1, dtype=int)
        self.potential_scores = np.zeros(13, dtype=int)
        self.rounds_left = 13
        self.rolls_this_round = 0
        self.update_potential_scores()
        return self.observation(), {}

    def step(self, action):
        """
        Execute the given action in the environment.

        Parameters
        ----------
        action : dict
            Contains 'dice_action' which is a MultiBinary(5) for dice rerolls and
            'score_action' which is a Discrete(13) to select a scoring category.

        Returns
        -------
        tuple
            Observations, reward, done (boolean), truncated (boolean), and info (dict).
        """
        dice_action = action["dice_action"]
        score_action = action["score_action"]
        assert self.action_space.contains(action), "Invalid action"

        if self.rolls_this_round < 2 and any(dice_action):
            self.reroll_dice(dice_action)
            self.rolls_this_round += 1
            self.update_potential_scores()
            return self.observation(), 0, False, False, {}

        # Enforce valid action, penalty inherent in suboptimal choice
        if score_action != -1 and self.scorecard[score_action] != -1:
            score_action = self.find_first_unscored_category()

        reward = self.score_category(score_action)
        self.rounds_left -= 1
        terminated = self.rounds_left == 0
        truncated = False

        self.dice = np.random.randint(1, 7, size=(5,))
        self.rolls_this_round = 0
        self.update_potential_scores()

        return self.observation(), reward, terminated, truncated, {}

    def find_first_unscored_category(self):
        """
        Finds the first category in the scorecard that has not been scored yet.

        Returns
        -------
        int
            The index of the first unscored category in the scorecard, or -1 if all categories have been scored.
        """
        for i in range(13):
            if self.scorecard[i] == -1:
                return i
        return -1

    def observation(self):
        """
        Constructs the current observation of the environment.

        Returns
        -------
        dict
            A dictionary containing:
            - 'dice': array of current dice values.
            - 'scorecard': current score status for each category.
            - 'potential_scores': potential score for each category based on current dice.
            - 'remaining_rolls': number of remaining rolls in this turn.
        """
        return {
            "dice": self.dice,
            "scorecard": self.scorecard,
            "potential_scores": self.potential_scores,
            "remaining_rolls": 3 - self.rolls_this_round,
        }

    def reroll_dice(self, dice_action):
        """
        Rerolls the dice specified by the dice_action.

        Parameters
        ----------
        dice_action : ndarray
            An array indicating which dice to reroll (1 to reroll, 0 to keep).
        """
        for i in range(5):
            if dice_action[i] == 1:
                self.dice[i] = np.random.randint(1, 7)

    def score_category(self, category):
        """
        Scores the specified category if it is unscored.

        Parameters
        ----------
        category : int
            The category index to score.

        Returns
        -------
        int
            The score for the selected category, or a penalty for invalid selection.
        """
        if self.scorecard[category] == -1:
            self.scorecard[category] = self.potential_scores[category]
            reward = self.potential_scores[category]
            self.potential_scores[category] = 0
            return reward
        return (
            -10
        )  # Potentially penalize invalid choice rather than enforce valid choice...

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
            print("Potential Scores:", self.potential_scores)
            print("Rolls left this round: ", 3 - self.rolls_this_round)
        elif mode == "ansi":
            return f"Dice: {self.dice} \nScorecard: {self.scorecard} \nRolls left this round: {3 - self.rolls_this_round}"

    def update_potential_scores(self):
        """Updates the potential scores for each category based on the current dice configuration."""
        counts = np.bincount(self.dice, minlength=7)[
            1:
        ]  # Count occurrences of each die face

        # Upper Section: Ones to Sixes
        for i in range(6):
            if self.scorecard[i] == -1:  # Check if not scored
                self.potential_scores[i] = (i + 1) * counts[i]

        # Three of a Kind
        self.potential_scores[6] = (
            np.sum(self.dice) if np.max(counts) >= 3 and self.scorecard[6] == -1 else 0
        )

        # Four of a Kind
        self.potential_scores[7] = (
            np.sum(self.dice) if np.max(counts) >= 4 and self.scorecard[7] == -1 else 0
        )

        # Full House
        self.potential_scores[8] = (
            25
            if (np.any(counts == 3) and np.any(counts == 2)) and self.scorecard[8] == -1
            else 0
        )

        # Small Straight
        self.potential_scores[9] = (
            30 if self.check_straight(counts, 4) and self.scorecard[9] == -1 else 0
        )

        # Large Straight
        self.potential_scores[10] = (
            40 if self.check_straight(counts, 5) and self.scorecard[10] == -1 else 0
        )

        # Yahtzee
        self.potential_scores[11] = (
            50 if np.max(counts) == 5 and self.scorecard[11] == -1 else 0
        )

        # Chance
        self.potential_scores[12] = np.sum(self.dice) if self.scorecard[12] == -1 else 0

    def check_straight(self, counts, length):
        """Helper function to check if there's a straight of a given length."""
        for i in range(1, 7 - length + 1):
            if all(counts[i : i + length] > 0):
                return True
        return False

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
            score_display = "-" if self.scorecard[i] == -1 else str(self.scorecard[i])
            print(f"| {title:<16} | {score_display:<5} |")
        print("+-------------------+-------+")

    def get_total_score(self):
        """
        Calculates and returns the total score.

        Returns
        -------
        int
            The total score calculated from the scorecard.
        """
        total_score = sum(self.scorecard)
        print("Total Score:", total_score)
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
