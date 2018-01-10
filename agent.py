import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import numpy as np
from collections import deque
import pickle

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model, save_model

class Agent:
    def __init__(self, state_sample, action_sample, epsilon=1.0, density=24, max_mem_size=1000):

        if not os.path.isdir('save'):
            os.mkdir('save')

        self.state_size = state_sample.shape[0]*state_sample.shape[1]
        self.action_size = action_sample.shape[0]*action_sample.shape[1]
        self.memory = deque(maxlen=max_mem_size)
        self.short_term_memory = deque(maxlen=max_mem_size)
        self.gamma = 0.7    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model(density=density)

    def _build_model(self, density=24):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(density, input_dim=self.state_size, activation='relu'))
        model.add(Dense(density, activation='relu'))
        model.add(Dense(self.action_size, activation='relu'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def clear_short_term_memory(self):
        self.short_term_memory.clear()

    def memory_size(self):
        return len(self.memory)

    def move_short_to_long_term_memory(self):
        self.memory.extend(self.short_term_memory)
        self.short_term_memory.clear()

    def remember(self, state, action, reward, next_state, done):
        self.short_term_memory.append((state, action, reward, next_state, done))

    def act(self, state, env):
        if np.random.rand() <= self.epsilon:
            rand_action = env.random_action()
            return rand_action

        state = np.copy(state).reshape(1, self.state_size)
        act_values = self.model.predict(state)
        action = env.action_vector_to_matrix(act_values)
        return action  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        fit_states = None
        fit_targets = None

        for state, action, reward, next_state, done in minibatch:

            target = reward

            if not done:
                # expected future reward for (state, action) combination
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))

            # model returns an action vector, with the predicted reward for each possible action
            target_f = self.model.predict(state)
            action_idx = np.argmax(action)
            target_f = target_f * 0.0
            target_f[0][action_idx] = target

            if fit_states is None:
                fit_states = state
            else:
                fit_states = np.vstack((fit_states, state))

            if fit_targets is None:
                fit_targets = target_f
            else:
                fit_targets = np.vstack((fit_targets, target_f))

        if fit_states.shape[0]>0:
            self.model.fit(fit_states, fit_targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, env):
        model_fname = './save/agent_' + env.name() + '.h5'

        if os.path.isfile(model_fname):
            self.model = load_model(model_fname)

        agent_epsilon_fname = './save/agent_' + env.name() + '_epsilon.p'

        if os.path.isfile(agent_epsilon_fname):
            self.epsilon = pickle.load(open(agent_epsilon_fname, "rb"))

    def save(self, env):
        model_fname = './save/agent_' + env.name() + '.h5'
        self.model.save(model_fname)

        agent_epsilon_fname = './save/agent_' + env.name() + '_epsilon.p'
        pickle.dump(self.epsilon, open(agent_epsilon_fname, "wb"))

