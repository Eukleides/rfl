import connect_four_gym
import random
import os
import time
import numpy as np
import psutil
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Flatten, Input, multiply
from replay_buffer import ReplayBuffer
GPU_VERSION = False

###################################
if GPU_VERSION:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
###################################

DISCOUNT_FACTOR_GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 128
TARGET_UPDATE_EVERY = 1000
TRAIN_START = 2000
REPLAY_BUFFER_SIZE = 50000
MAX_STEPS = 200000
LOG_EVERY = 2000
SNAPSHOT_EVERY = 50000
EVAL_EVERY = 20000
EVAL_STEPS = 100
EVAL_EPSILON = 0
TRAIN_EPSILON = 0.01
Q_VALIDATION_SIZE = 10000

def one_hot_encode(n, action):
    one_hot = np.zeros(n)
    one_hot[int(action)] = 1
    return one_hot

def predict(env, model, observations):
    mask = np.ones((len(observations), env.action_space.n))
    return model.predict(x=[observations, mask])

def fit_batch(env, model, target_model, batch):
    observations, actions, rewards, next_observations, dones, players = batch
    # Predict the Q values of the next states. Passing ones as the action mask.
    next_q_values = predict(env, target_model, next_observations)
    # The next Q values of terminal states are 0 by definition.
    next_q_values[dones] = 0.0
    # The Q values of each start state is the reward + gamma * the max next state Q value
    q_values = rewards - DISCOUNT_FACTOR_GAMMA * np.max(next_q_values, axis=1)
    one_hot_actions = np.array([one_hot_encode(env.action_space.n, action) for action in actions])
    history = model.fit(
        x=[observations, one_hot_actions],
        y=one_hot_actions * q_values[:, None],
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    return history.history['loss'][0]

def create_model(env):
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    input_tensor = Input(shape=obs_shape)
    action_mask = Input(shape=(n_actions,))

    f1 = Flatten()(input_tensor)
    d1 = Dense(32, activation='relu', name='dense1')(f1)
    d2 = Dense(32, activation='relu', name='dense2')(d1)
    y0 = Dense(n_actions, activation='linear', name='y0')(d2)
    y = multiply([y0, action_mask])

    model = Model(inputs=[input_tensor, action_mask], outputs=[y])
    optimizer = keras.optimizers.Adam(lr=LEARNING_RATE, clipnorm=1.0)
    model.compile(optimizer, loss='mean_squared_error')

    return model

def greedy_action(env, model, observation):
    next_q_values = predict(env, model, observations=observation)
    action = np.argmax(next_q_values)
    if env.actionIsValid(action):
        return action

    # action is invalid, pick up next most recommended action
    minVal = np.min(next_q_values)-1.0
    for i in range(len(next_q_values[0])):
        if env.actionIsValid(i) is False:
            next_q_values[0][i] = minVal
    action = np.argmax(next_q_values)
    return action

def epsilon_greedy_action(env, model, opponentModel, observation, epsilon):

    if env.isOpponentsTurn():
        action = greedy_action(env, opponentModel, observation)
        return action

    if random.random() < epsilon:
        action = env.getValidRandomAction()
    else:
        action = greedy_action(env, model, observation)
    return action

def save_model(model, step, name):
    if name is None:
        name = 'model'
    filename = 'models/{}-{}.h5'.format(name, step)
    model.save(filename)
    print('Saved {}'.format(filename))
    return filename

def evaluate(env, model, opponentModel=None, view=False, numGames=100):

    print("Evaluation")
    done = False
    episode = 0
    episode_win_return_sum = 0.0
    episode_lose_return_sum = 0.0
    obs = env.reset()
    is_ai_player = None
    reward = 0.

    while episode<numGames:

        if done:
            if is_ai_player is True:
                episode_win_return_sum += reward
            elif is_ai_player is False:
                episode_lose_return_sum += reward

            if view:
                env.render()
                time.sleep(1.0)

            obs = env.reset()
            episode += 1

        action = epsilon_greedy_action(env, model, opponentModel, obs, epsilon=EVAL_EPSILON)
        obs, reward, done, player = env.step(action)
        is_ai_player = (player == env.metadata['AI PLAYER'])


    avg_win_return = episode_win_return_sum/episode
    avg_lose_return = episode_lose_return_sum / episode

    return avg_win_return, avg_lose_return

def train(env, model, opponentModel, max_steps, name):

    target_model = create_model(env)
    replay = ReplayBuffer(REPLAY_BUFFER_SIZE)
    done = True
    episode = 0
    steps_after_logging = 0
    loss = 0.0

    for step in range(1, max_steps + 1):
        try:
            if step % SNAPSHOT_EVERY == 0:
                save_model(model, step, name)

            if done:

                if episode > 0:

                    if steps_after_logging >= LOG_EVERY:

                        steps_after_logging = 0
                        episode_end = time.time()
                        episode_seconds = episode_end - episode_start
                        episode_steps = step - episode_start_step
                        steps_per_second = episode_steps / episode_seconds
                        memory = psutil.virtual_memory()
                        to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024

                        print(
                            "episode {} "
                            "steps {}/{} "
                            "loss {:.7f} "
                            "return {} "
                            "in {:.2f}s "
                            "{:.1f} steps/s "
                            "{:.1f}/{:.1f} GB RAM".format(
                                episode,
                                episode_steps,
                                step,
                                loss,
                                episode_return,
                                episode_seconds,
                                steps_per_second,
                                to_gb(memory.used),
                                to_gb(memory.total),
                            ))

                # env.render()
                episode_start = time.time()
                episode_start_step = step
                obs = env.reset()
                episode += 1
                episode_return = 0.0

            else:
                obs = next_obs

            action = epsilon_greedy_action(env, model, opponentModel, obs, epsilon=TRAIN_EPSILON)
            next_obs, reward, done, player = env.step(action)

            if player == env.metadata['AI PLAYER']:
                episode_return += reward
            else:
                episode_return -= reward

            replay.add(obs[0], action, reward, next_obs[0], done, player)

            if step >= TRAIN_START:
                if step % TARGET_UPDATE_EVERY == 0:
                    target_model.set_weights(model.get_weights())
                batch = replay.sample(BATCH_SIZE, choose_players=[env.metadata['AI PLAYER']])
                loss = fit_batch(env, model, target_model, batch)

                if step % EVAL_EVERY == 0:
                    avg_win_return, avg_lose_return = evaluate(env, model, opponentModel)

                    print(
                        "avg_win_return {:.1f} , avg_lose_return {:.1f} ".format(
                            avg_win_return,
                            avg_lose_return
                            ))

            steps_after_logging += 1

        except KeyboardInterrupt:
            save_model(model, step, name)
            break

def load_or_create_model(env, model_filename):
    if model_filename:
        fullname = 'models/{}.h5'.format(model_filename)
        model = keras.models.load_model(fullname)
        print('Loaded {}'.format(fullname))
    else:
        model = create_model(env)
    return model

def set_seed(env, seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    env.seed(seed)

def main(play=False, model_name=None):

    env = connect_four_gym.FooEnv()
    set_seed(env, 0)
    model = load_or_create_model(env, model_name)
    opponentModel = load_or_create_model(env, 'opponent')

    if play:
        episode_win_return_sum, episode_lose_return_sum = evaluate(env, model, opponentModel, view=True, numGames=50)
        print("Wins:{:.1f} Loses:{:.1f}".format(episode_win_return_sum, episode_lose_return_sum))
        env.close()
    else:
        max_steps = 120000
        train(env, model, opponentModel, max_steps, None)
        if True:
            filename = save_model(model, EVAL_STEPS, name='test')

main(play=True, model_name='test-100')

