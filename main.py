import gym
import numpy as np
# from keras.utils import to_categorical
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.models import load_model
# import random

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers

def run_episode(env, Q, learning_rate, discount, episode, render=False):

    state = env.reset()
    done = False

    while not done:

        if render:
            env.render()

        action = np.argmax(Q[state, :])
        if np.random.random_sample()<0.50:
            action = env.action_space.sample()

        next_state, reward, done, info = env.step(action)

        if done:
            Q[state, action] = reward
        else:
            Q[state, action] = reward + discount * np.max(Q[next_state, :])

        state = next_state

    return Q


def train():
    env = gym.make('FrozenLake-v0')
    learning_rate = 0.81
    discount = 0.96
    num_episodes = 10000

    Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=float)

    for i in range(num_episodes):
        Q = run_episode(env, Q, learning_rate, discount, i, render=False)

    # run_episode(env, Q, learning_rate, discount, 100000000, render=True)
    return Q


q = train()
q *= 100.
np.set_printoptions(precision=1)
print(q)
