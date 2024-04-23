# Blackjack Monte Carlo Agent

This Python project implements a Blackjack-playing agent using the Monte Carlo method for reinforcement learning. The agent learns to optimize its strategy based on experience gained from playing multiple episodes of Blackjack, utilizing an epsilon-greedy policy for action selection.

## Project Description

The agent operates within the `gymnasium` Blackjack environment. Through repeated play, it adjusts its strategy by estimating the value of state-action pairs based solely on the outcomes of completed episodes. This model-free approach allows the agent to improve its decisions over time without a predefined model of the environment.

## Features

- **Monte Carlo Method:** Utilizes the Monte Carlo method for learning the value of state-action pairs and determining policy based on those estimates.
- **Epsilon-Greedy Policy:** Balances exploration of new actions with exploitation of known strategies to refine the agent’s decisions.
- **Statistical Outcome Visualization:** Uses `matplotlib` to visualize the distribution of wins, draws, and losses after the agent has completed a predefined number of episodes.

## Requirements

This project requires the following Python libraries:

- `numpy`
- `gymnasium`
- `matplotlib`

Install these with pip using the following command:

```bash
pip install numpy gymnasium matplotlib
```

## Usage

```bash
python main.py
```

This initiates a sequence of Blackjack games, with the agent learning and adjusting its policy after each game. The number of episodes can be modified by changing the n_episodes variable in the script.
