
# Blackjack Agent with Monte Carlo Learning

This Python implementation uses the Monte Carlo first-visit prediction method to learn the state value function for a game of Blackjack. The code utilizes the OpenAI `gymnasium` environment to simulate the Blackjack game, wherein an agent attempts to learn the optimal policy for the game based on historical data from each playthrough.

## Features

- State Value Estimation: The agent estimates the value of each state using the first-visit Monte Carlo method.
- Simple Policy Implementation: The policy is to stand if the hand total is 20 or 21 and hit otherwise.
- Flexible Gamma Value: The discount factor (gamma) can be adjusted at the time of agent initialization.

## Requirements

To run this script, you need the following libraries:  

- `numpy`
- `gymnasium`

You can install them using pip:

```bash
pip install numpy gymnasium
```

## Usage

To run the simulation, execute the main function in the provided script. The simulation runs through a specified number of episodes, updating the agent's understanding of state values throughout the game. You can adjust the number of episodes by changing the n_episodes variable in the main function.

```bash
python main.py
```
