import gym
from Agent import *
import matplotlib.pyplot as plt

# Configuration parameters
batch_size = 32
gamma = 0.9
train_episodes = 300
max_epochs = 100
n_test_trials = 10

# Create environment
env = gym.make('CartPole-v1')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = Agent(state_size, action_size)

rewards = agent.train(env, train_episodes, batch_size)
best_ind = np.argmax(rewards)
print("Best Score was: {} in Episode: {}".format(rewards[best_ind], best_ind))

plt.plot(rewards)
plt.title("Rewards over time; average score: " + str(sum(rewards) / train_episodes))
plt.show()


agent.test(env, n_test_trials)
