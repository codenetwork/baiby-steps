import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stable_baselines3 import DDPG, PPO
from envs.simple_biped_env import simpleBipedEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium.envs.registration import register

from simple_biped_env import simpleBipedEnv

register(
    id='simpleBiped-v0',
    entry_point='envs.simple_biped_env:simpleBipedEnv',
    max_episode_steps=10000,
)

def train_PPO():
    LOG_DIR = "./logs/"
    env = gym.make('simpleBiped-v0', render=True)
    model = PPO("MlpPolicy", env, verbose=1, device="auto", tensorboard_log=LOG_DIR)
    model.learn(total_timesteps=500000)
    model.save("../models/humanoid_ppo")
    env.close()

def train_DDPG():
    LOG_DIR = "./logs/"
    env = gym.make('simpleBiped-v0', render=True)
    model = DDPG("MlpPolicy", env, verbose=1, device="auto", tensorboard_log=LOG_DIR)
    model.learn(total_timesteps=10000000)
    model.save("../models/humanoid_ddpg")
    env.close()

def run():
    env = DummyVecEnv([lambda: simpleBipedEnv(render=True)])
    model = PPO.load("models/humanoid_ppo", env=env)
    for _ in range(5):  # Run 3 episodes
        result = env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        print(f"Episode finished. Total reward: {total_reward}")
    env.close()

def test():
    env = simpleBipedEnv(render=True)
    for _ in range(10):  # Run 1 test episode
        obs = env.reset()[0]
        done = False
        total_reward = 0.0
        while not done:
            action = env.action_space.sample()  # Random action for testing
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        print(f"Test episode finished. Total reward: {total_reward}")
    env.close()

train_PPO()
#train_DDPG()
#run()
#test()
