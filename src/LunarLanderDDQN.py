import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

import csv

import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from algorithms.DoubleDQN import DoubleDQN

for i in range(5):
    print(f"Training model {i+1}")
    env = gym.make('LunarLander-v2')
    # env.seed(0)
    np.random.seed(0)

    print('State space: ', env.observation_space)
    print('Action space: ', env.action_space)

    model = keras.Sequential()
    model.add(keras.layers.Dense(512, activation="relu", input_dim=env.observation_space.shape[0]))
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(env.action_space.n, activation="linear"))
    model.summary()

    gamma = 0.75
    epsilon = 1.0 
    epsilon_min = 0.01
    epsilon_dec = 0.99
    episodes = 500
    learning_rate = 0.001
    batch_size = 64
    memory = deque(maxlen=10000) 
    max_steps = 1500
    interval = 2

    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

    DDQN = DoubleDQN(env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory, model, max_steps, interval)
    rewards = DDQN.train()

    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('# Rewards')
    plt.title('# Rewards vs Episodes')
    plt.savefig(f"results/lunar_lander_DDQN {i+1}.jpg")     
    plt.close()

    with open(f'results/lunar_lander_DDQN_rewards {i+1}.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        episode=0
        for reward in rewards:
            writer.writerow([episode,reward])
            episode+=1

    model.save(f"data/model_DDQN_lunar_lander{i+1}.keras")
