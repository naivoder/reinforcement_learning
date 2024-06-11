# CartPole Q-Learning Agent

This project implements a Q-learning reinforcement learning agent to solve the CartPole problem in the Gymnasium environment. It features a discretization strategy for the continuous state space and an epsilon-greedy policy for action selection. The agent learns optimal actions over time to balance a pole on a moving cart.

## Features

- **Q-Learning**: Implementation of the Q-learning algorithm for value-based learning.
- **Epsilon-Greedy Policy**: Action selection using an epsilon-greedy approach to balance exploration and exploitation.
- **State Discretization**: Converts continuous state variables into discrete bins to manage the Q-table.
- **Visualization**: Plots the running average of the rewards to demonstrate learning progress over episodes.

## Requirements

To run this script, you need the following libraries:  

- `numpy`
- `gymnasium`
- `matplotlib`

You can install them using pip:

```bash
pip install numpy gymnasium matplotlib
```

## Usage

To run the Cartpole TD-Learning agent, execute the main Python script:

```bash
python main.py
```

The script will initiate a series of 50,000 episodes where the agent learns to balance the pole on the cart. Output will be printed showing the progress of episodes and a plot of the running average of scores will be generated.
