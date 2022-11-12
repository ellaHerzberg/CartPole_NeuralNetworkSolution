import keras
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.alpha = 0.001
        self.gamma = 0.95
        self.epsilon = 1.0

        self.memory = deque(maxlen=2000)
        self.model = self._build_model()

    def _build_model(self):
        """
        This method creates the agent's model.
        """
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha))
        return model

    def _act(self, state):
        """
        This method choose the agent's action.
        """
        # Exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # Exploitation
        actions = self.model.predict(state)
        return np.argmax(actions[0])

    def _update_memory(self, state, action, reward, next_state, done):
        """
        The function update the agent's memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def _learn(self, batch_size):
        # If the memory is not big enough
        if len(self.memory) < batch_size:
            return
        # Sample from the memory
        sample_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in sample_batch:
            est_reward = reward
            if not done:
                # learning role:
                next_state = next_state.reshape(1, -1)
                prediction = self.model.predict(next_state)
                predicted_reward = np.amax(prediction[0])
                est_reward = reward + self.gamma * predicted_reward
            state = state.reshape(1, -1)
            curr_value = self.model.predict(state)
            curr_value[0][action] = est_reward
            # Update the model
            self.model.fit(state, curr_value, epochs=1, verbose=0)
            self._update_epsilon()

    def _update_epsilon(self):
        """
        Minimize epsilon as the learning progress
        """
        epsilon_min = 0.01
        epsilon_decay = 0.995

        if self.epsilon > epsilon_min:
            self.epsilon = self.epsilon * epsilon_decay

    def _save_model(self):
        self.model.save("CartPole_model_3.h5")

    def _load_model(self):
        model = keras.models.load_model("CartPole_model_3.h5")
        return model

    def train(self, env, train_episodes, batch_size):
        rewards = []
        for episode in range(train_episodes):
            state = env.reset()
            state = state.reshape(1, -1)

            done = False
            indx = 0
            while not done:
                action = self._act(state)

                next_state, reward, done, _ = env.step(action)
                next_state = next_state.reshape(1, -1)
                if done:
                    reward = -1
                self._update_memory(state, action, reward, next_state, done)
                state = next_state
                indx += 1
            print("-----------------------------------------------------Episode {}# Score: {}".format(episode, indx + 1))
            rewards.append(indx)
            self._learn(batch_size)
        print("Score over time: " + str(sum(rewards) / train_episodes))  # optional
        self._save_model()
        return rewards

    def test(self, env, n_test_trials):
        self.model = self._load_model()
        rewards = []
        for trial in range(n_test_trials):
            state = env.reset()
            state = state.reshape(1, -1)
            indx = 0
            done = False

            print("****************************************************")
            print("TRIAL ", trial)
            while not done:
                actions = self.model.predict(state)
                action = np.argmax(actions[0])
                env.render()
                # time.sleep(0.05)

                next_state, reward, done, _ = env.step(action)
                indx += 1
                if done:
                    print("-------------------------------------------------Trial {}#, Score: {}".format(trial, indx+1))
                state = next_state.reshape(1, -1)
            rewards.append(indx)
        print("Score over time: " + str(sum(rewards) / n_test_trials))  # optional
        env.close()
