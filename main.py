import gym
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
###################################

seed = 7
np.random.seed(seed)
env = gym.make("CartPole-v0")
env.reset()

def play_a_game(model=None, render=False, verbose=False):
    # initialise
    total_reward = 0.
    observation = env.reset()

    # holders to record observations, actions
    obs_hist = []
    act_hist = []

    done = False

    # this is each frame, up to 200...but we wont make it that far.
    while not done:
        # This will display the environment
        # Only display if you really want to see it.
        # Takes much longer to display it.
        if render:
            env.render()

        # add observation to history
        obs_hist.append(observation)

        # This will just create a sample action in any environment.
        # In this environment, the action can be 0 or 1, which is left or right
        if model is None:
            action = env.action_space.sample()
        else:
            # imply action from the model
            np_obs = observation.reshape(1,-1)
            action_oh = model.predict(np_obs)
            action = np.argmax(action_oh)

        # this executes the environment with an action,
        # and returns the observation of the environment,
        # the reward, if the env is over, and other info.
        observation, reward, done, info = env.step(action)

        total_reward += reward

        # add action to history
        act_hist.append([action])

        if done:
            break

    if verbose:
        print('Total Reward: ' + '{:.0f}'.format(total_reward))

    obs_hist = np.array(obs_hist, dtype=float)
    act_hist = np.array(act_hist, dtype=float)
    act_hist.reshape(act_hist.size,1)
    return total_reward, obs_hist, act_hist

def get_some_winning_games(ngames, min_score):

    nrec = 0
    obs_hist = np.array([])
    act_hist = np.array([])

    while(nrec<ngames):
        reward, obs_i, act_i = play_a_game()

        if reward >= min_score:
            # this was a good game, store it
            if obs_hist.size==0:
                obs_hist = obs_i
                act_hist = act_i
            else:
                obs_hist = np.vstack((obs_hist, obs_i))
                act_hist = np.vstack((act_hist, act_i))
            nrec += 1

    # one hot encode
    act_hist_oh = to_categorical(act_hist)

    return obs_hist, act_hist_oh

def train_model(obs_hist, act_hist):
    # create model
    model = Sequential()
    model.add(Dense(64, input_dim=4, activation='relu'))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(obs_hist, act_hist, epochs=250, batch_size=10)

    # save the trained model
    model.save('my_model.h5')

    return model

# obs_hist, act_hist = get_some_winning_games(ngames=10, min_score=50.)
# model = train_model(obs_hist=obs_hist, act_hist=act_hist)
#
model = load_model('my_model.h5')
# for _ in range(10):
#     obs_hist, act_hist = get_some_winning_games(ngames=10, min_score=50.)
#     model = train_model(obs_hist=obs_hist, act_hist=act_hist)

play_a_game(model=model, render=True, verbose=True)