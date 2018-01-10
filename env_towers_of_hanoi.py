from env import Env
import numpy as np
from agent import Agent

# Towers of Hanoi environment
class TowersEnv(Env):

    def __init__(self):
        self.reset()

    def reset(self):
        num_blocks = 3
        num_towers = 3
        state1 = np.ones((num_blocks, num_towers)) * (-1.0)
        state1[2][0] = 2
        state1[1][0] = 1
        state1[0][0] = 0
        Env.__init__(self, state=state1, env_name='Towers of Hanoi')

        return np.copy(self.state)

    def num_blocks(self, pos):
        count = self.state.shape[0]
        for i in range(self.state.shape[0]):
            if self.state[i][pos] != -1:
                return count
            count -= 1

        return count

    def pop_block(self, pos):
        for i in range(self.state.shape[0]):
            if self.state[i][pos] != -1:
                v = self.state[i][pos]
                self.state[i][pos] = -1
                return v
        raise Exception('not meant to get here!')

    def push_block(self, pos, value):
        for i in range(self.state.shape[0]-1, -1, -1):
            if self.state[i][pos] == -1:
                self.state[i][pos] = value
                return
        raise Exception('not meant to get here!')

    def random_action(self):
        l = []
        for i in range(self.state.shape[1]):
            if self.num_blocks(i)>0:
                l.append(i)

        f = np.random.choice(l)
        l = list(range(self.state.shape[1]))
        l.remove(f)
        t = np.random.choice(l)

        a = np.zeros((self.state.shape[1], self.state.shape[1]))
        a[f, t] = 1.0
        return a

    def action_to_from_to(self, action):
        f, t = np.unravel_index(action.argmax(), action.shape)

        return f, t

    def apply_action(self, f, t):
        v = self.pop_block(f)
        self.push_block(pos=t, value=v)

    def get_score(self):
        score = 0

        for i in range(self.state.shape[0]-1, -1, -1):
            if self.state[i][self.state.shape[1]-1] == i:
                score += 10.0

        return score

    def is_game_over(self):
        count = 0

        for i in range(self.state.shape[0]-1, -1, -1):
            if self.state[i][self.state.shape[1]-1] == i:
                count += 1.0

        return count == self.state.shape[0]

    def step(self, action):
        f, t = self.action_to_from_to(action)
        ante_score = self.get_score()

        if self.num_blocks(f)==0:
            post_score = self.get_score()
            reward = post_score - ante_score
            return self.state, reward, False, 'invalid move'

        self.apply_action(f, t)
        post_score = self.get_score()
        reward = post_score - ante_score
        next_state = np.copy(self.state)
        done = self.is_game_over()
        info = ''

        return next_state, reward, done, info

def play(env):
    agent = Agent(env.state, env.random_action())
    agent.epsilon = 0.0
    agent.load(env)

    for i_episode in range(2):

        state = env.reset()
        print('-------------------------------')
        print(state)

        for t in range(10):

            action = agent.act(state, env)
            state, reward, done, info = env.step(action)

            print()
            print(state)

            if done:
                print('Episode finished after {} timesteps'.format(t+1))
                break

