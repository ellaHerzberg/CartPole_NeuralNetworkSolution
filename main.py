import gym
from Agent import *

# Configuration parameters
batch_size = 32
n_train_episodes = 300
max_epochs = 100
n_test_trials = 10

model_name = "CartPole_model.h5"

# Create environment
env = gym.make('CartPole-v1')

# Extract parameters
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Create the agent
agent = Agent(state_size, action_size)

# Train
# train_rewards = agent.train(env, n_train_episodes, batch_size, model_name)
# agent.plot_learning(train_rewards)

# Test
test_rewards = agent.test(env, n_test_trials, model_name)
agent.plot_learning(test_rewards)
