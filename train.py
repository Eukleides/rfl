import numpy as np
from agent import Agent

def train(env, density=128, episodes=10000, time_threshold=10, max_time=200):

    agent = Agent(env.state, env.random_action(), density=density)
    agent.load(env)

    for e in range(episodes):

        state = env.reset()
        agent.clear_short_term_memory()

        for time in range(max_time):

            action = agent.act(state, env)
            next_state, reward, done, _ = env.step(action)

            agent.remember(state.reshape(1,-1), action.reshape(1,-1), reward, next_state.reshape(1,-1), done)
            state = np.copy(next_state)

            if done:
                print("episode: {}/{}, score: {}, time:{}, e: {:.2}"
                      .format(e, episodes, env.get_score(), time, agent.epsilon))
                break

        if time<time_threshold:

            agent.move_short_to_long_term_memory()

            if agent.memory_size()>100:
                agent.replay(agent.memory_size())

        if e % 10 == 0:
            agent.save(env)

