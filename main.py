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
env._max_episode_steps = 2000
env.reset()

# plays a single game
# model - a keras model to output actions
# render - display game screen (slow)
def play_a_game(model=None, render=False, verbose=False, random_move_chance=None):
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
            random_move = False

            # even though a model has been supplied, there is a random_move_chance
            if random_move_chance is not None:
                # there is a random_move_chance that we take a random action
                r = random.randint(1,101)
                rlim = int(float(random_move_chance*100.))
                if r <= rlim:
                    random_move = True

            if random_move:
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

# play until reached ngames of score at least min_score
# return (observations, actions) during those games
def get_some_winning_games(ngames, min_score, model=None, random_move_chance=None):

    nrec = 0
    obs_hist = np.array([])
    act_hist = np.array([])

    while(nrec<ngames):
        reward, obs_i, act_i = play_a_game(model=model, random_move_chance=random_move_chance)

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

# play ngames, return the top_pct expected score
def get_expected_top_pct_reward(ngames=50, top_pct=0.1, model=None, verbose=False, random_move_chance=None):
    nplayed = 0
    rewards = []

    while(nplayed<ngames):
        reward, _, _ = play_a_game(random_move_chance=random_move_chance, model=model, verbose=verbose)
        rewards.append(reward)
        nplayed += 1

    rewards.sort(reverse=True)
    top_idx = int(len(rewards)*float(top_pct))
    return rewards[top_idx]

def train_model(obs_hist, act_hist):
    # create model
    model = Sequential()
    model.add(Dense(128, input_dim=4, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(obs_hist, act_hist, epochs=5, batch_size=50, verbose=1)

    # save the trained model
    model.save('my_model.h5')

    return model

# min_score = get_expected_top_pct_reward()
# obs_hist, act_hist = get_some_winning_games(ngames=100, min_score=min_score)
# model = train_model(obs_hist=obs_hist, act_hist=act_hist)
#
model = load_model('my_model.h5')
rmove_chance = 0.5
for _ in range(20):
    min_score = get_expected_top_pct_reward(model=model, random_move_chance=rmove_chance)
    print('Top pct score: ' + '{:.0f}'.format(min_score))
    obs_hist, act_hist = get_some_winning_games(ngames=100, min_score=min_score, model=model)
    model = train_model(obs_hist=obs_hist, act_hist=act_hist)
    rmove_chance *= 0.9

# model = load_model('my_model.h5')
# play_a_game(model=model, render=True, verbose=True)
