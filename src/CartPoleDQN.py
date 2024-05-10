import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

import csv

import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from algorithms.DeepQNetworks import DeepQNetworks

print(f'{os.environ["KERAS_BACKEND"]}')

for i in range(5):
    print(f"Training model {i+1}")
    env = gym.make('CartPole-v1')
    #env.seed(0)
    np.random.seed(0)

    print('State space: ', env.observation_space)
    print('Action space: ', env.action_space)

    value_model = keras.Sequential()
    value_model.add(keras.layers.Dense(512, activation="relu", input_dim=env.observation_space.shape[0]))
    value_model.add(keras.layers.Dense(256, activation="relu"))
    value_model.add(keras.layers.Dense(128, activation="relu"))
    value_model.add(keras.layers.Dense(env.action_space.n, activation="linear"))

    value_model.summary()

    gamma = 0.99
    epsilon = 1.0 
    epsilon_min = 0.01
    epsilon_dec = 0.99
    learning_rate = 0.001
    episodes = 300
    batch_size = 64
    memory = deque(maxlen=10000)
    max_steps = 1500
    interval = 2

    value_model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

    DQN = DeepQNetworks(env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory, value_model, max_steps, interval)
    rewards = DQN.train()

    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('# Rewards')
    plt.title('# Rewards vs Episodes')
    plt.savefig(f"results/cart_pole_DQN_{i+1}.jpg")     
    plt.close()

    with open(f'results/cart_pole_DQN_rewards_{i+1}.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        episode=0
        for reward in rewards:
            writer.writerow([episode,reward])
            episode+=1

    value_model.save(f'data/model_DQN_cart_pole_{i+1}.keras')

