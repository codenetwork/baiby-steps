import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stable_baselines3 import PPO
from envs.walker_env import WalkerEnv
from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv([lambda: WalkerEnv(render=False)])
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)
model.save("models/humanoid_ppo")


env = WalkerEnv(render=True)
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)

input("Press Enter to exit and close the visualization window...")
