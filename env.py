from agent import Agent

# Base class for an environment
# inherited environments must support the following methods:
# reset() - initialise environment
# step(action) - apply action to the environment. Returns: next_state, reward, done, info
# get_score() - current score
# random_action() - random action
class Env():
    def __init__(self, state, env_name):
        self.state = state
        self.state_size = self.state.shape[0]*self.state.shape[1]
        self.action_size = self.state.shape[1] * self.state.shape[1]
        self.env_name = env_name

    def name(self):
        return self.env_name

    def action_vector_to_matrix(self, v):
        return v.reshape((self.state.shape[1], self.state.shape[1]))

    def state_vector_to_matrix(self, s):
        return s.reshape((self.state.shape[0], self.state.shape[1]))

    def action_matrix_to_vector(self, v):
        return v.reshape(-1)

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

