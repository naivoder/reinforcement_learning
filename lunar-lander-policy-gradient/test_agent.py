import pytest
import numpy as np
from agent import Agent


@pytest.fixture
def setup_agent():
    """Fixture to create an Agent with a typical configuration."""
    lr = 0.01
    input_dims = (8,)  # Assuming an 8-dimensional state space
    n_actions = 4
    gamma = 0.99
    agent = Agent(lr=lr, input_dims=input_dims, n_actions=n_actions, gamma=gamma)
    return agent


def test_agent_initialization(setup_agent):
    """Test initialization of Agent attributes."""
    agent = setup_agent
    assert agent.lr == 0.01
    assert agent.input_dims == (8,)
    assert agent.n_actions == 4
    assert agent.gamma == 0.99
    assert isinstance(agent.reward_memory, list)
    assert isinstance(agent.action_memory, list)
    assert len(agent.reward_memory) == 0
    assert len(agent.action_memory) == 0


def test_choose_action(setup_agent):
    """Test the choose_action method outputs an integer action."""
    agent = setup_agent
    state = np.random.rand(8)  # Random state from the environment
    action = agent.choose_action(state)
    assert isinstance(action, int)
    assert 0 <= action < agent.n_actions


def test_store_rewards(setup_agent):
    """Test storing rewards in the reward memory."""
    agent = setup_agent
    rewards = [10, -5, 0, 15.5]
    for reward in rewards:
        agent.store_rewards(reward)
    assert agent.reward_memory == rewards


def test_learning_process(setup_agent):
    """Test the learning process by ensuring memory is cleared after learning."""
    agent = setup_agent
    states = [np.random.rand(8) for _ in range(5)]
    rewards = [10, -5, 0, 15.5, -20]

    for state in states:
        agent.choose_action(state)

    for reward in rewards:
        agent.store_rewards(reward)

    assert len(agent.reward_memory) == 5
    assert len(agent.action_memory) == 5

    agent.learn()
    assert len(agent.reward_memory) == 0
    assert len(agent.action_memory) == 0
