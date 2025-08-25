import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stable_baselines3 import PPO
from envs.simple_biped_env import simpleBipedEnv
from stable_baselines3.common.vec_env import DummyVecEnv

def train():
    env = DummyVecEnv([lambda: simpleBipedEnv(render=False)])
    model = PPO("MlpPolicy", env, verbose=1 )
    model.learn(total_timesteps=100000)
    model.save("models/humanoid_ppo")
    env.close()

def run():
    env = DummyVecEnv([lambda: simpleBipedEnv(render=True)])
    model = PPO.load("models/humanoid_ppo", env=env)
    for _ in range(3):  # Run 3 episodes
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

#train()
run()
#test()
