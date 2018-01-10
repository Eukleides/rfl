import random
from env_towers_of_hanoi import TowersEnv
from train import train

if __name__ == "__main__":

    random.seed(1)

    env = TowersEnv()
    train(env, density=128, episodes=10000, time_threshold=10, max_time=200)
