import gym
from keras.models import load_model
import numpy as np
from utils import tf_memory

tf_memory()

STATE_SIZE = 4

env = gym.make('CartPole-v1')
env.reset()

model = load_model('cartpole_nes.h5')

state = env.reset()
state = np.reshape(state, [1, STATE_SIZE])

done = False

while not done:
    action = model.predict(state)[0]

    # comment this out if you're not doing CartPole or something
    action = np.argmax(action)
    env.render()
    new_state, reward, done, _ = env.step(action)
    if done:
        break
    state = new_state
    state = np.reshape(state, [1, STATE_SIZE])
