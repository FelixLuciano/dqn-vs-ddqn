import gymnasium as gym
import keras
import numpy as np

env = gym.make('LunarLander-v2', render_mode='human').env
(state,_) = env.reset()
model = keras.models.load_model('data/model_DDQN_lunar_lander.keras')
done = False
truncated = False
rewards = 0
steps = 0
max_steps = 500

while (not done) and (not truncated) and (steps<max_steps):
    Q_values = model.predict(state[np.newaxis], verbose=0)
    action = np.argmax(Q_values[0])
    state, reward, done, truncated, info = env.step(action)
    rewards += reward
    env.render()
    steps += 1

print(f'Score = {rewards}')
input('press a key...')