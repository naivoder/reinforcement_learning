# CartPole TD-Learning Agent

This project implements a Temporal Difference (TD) Learning agent to estimate state values in the CartPole environment from OpenAI's Gym library. The agent discretizes the pole angle into bins and updates value estimates using the TD(0) method. It aims to balance a pole on a cart by applying forces to the left or right.

## Requirements

To run this script, you need the following libraries:  

- `numpy`
- `gymnasium`

You can install them using pip:

```bash
pip install numpy gymnasium
```

## Usage

To run the Cartpole TD-Learning agent, execute the main Python script:

```bash
python main.py
```

The script will initiate a series of episodes where the agent learns to balance the pole on the cart. Output will be printed showing the progress of episodes and the final value estimates after training.
