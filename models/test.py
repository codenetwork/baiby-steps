import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stable_baselines3 import PPO
from envs.simple_biped_env import simpleBipedEnv
from stable_baselines3.common.vec_env import DummyVecEnv

def train():
    env = DummyVecEnv([lambda: simpleBipedEnv(render=False)])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000)
    model.save("models/humanoid_ppo")
    env.close()

def run():
    env = DummyVecEnv([lambda: simpleBipedEnv(render=True)])
    model = PPO.load("models/humanoid_ppo", env=env)

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

train()
#run()

input("Press Enter to exit and close the visualization window...")
