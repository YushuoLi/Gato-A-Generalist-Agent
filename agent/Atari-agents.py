import gym
import time
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv
from gym.wrappers import GrayScaleObservation

# 环境 Breakout-v4
env = gym.make('Breakout-v4')
env = EpisodicLifeEnv(env)


# 装饰器
monitor_dir = r'./monitor_log/'
from stable_baselines3.common.monitor import Monitor
env = Monitor(env, monitor_dir)
env = GrayScaleObservation(env,keep_dim=True)
print(env)
from stable_baselines3.common.vec_env import DummyVecEnv
env = DummyVecEnv([lambda: env])
state = env.reset()
print(state.shape)
from stable_baselines3.common.vec_env import VecFrameStack
env = VecFrameStack(env,4,channels_order='last')
state = env.reset()
print(state.shape)
from stable_baselines3.common.vec_env import VecTransposeImage
env = VecTransposeImage(env)
state = env.reset()
print(state.shape)

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
import os

# ----------------------------------------------------- 训练 --------------------------------------------
seed = 123
model = PPO(policy='CnnPolicy', env=env, verbose=1, seed=seed)
model.learn(total_timesteps=500000)
model.save("./models/atari_ppo")

model = DQN(policy='CnnPolicy', env=env, verbose=1, seed=seed)
model.learn(total_timesteps=500000)
model.save("./models/atari_dqn")

model = A2C(policy='CnnPolicy', env=env, verbose=1, seed=seed)
model.learn(total_timesteps=500000)
model.save("./models/atari_a2c")


# ----------------------------------------------------- 测试 --------------------------------------------
# model = PPO.load("./models/atari_ppo")
# obs = env.reset()
#
# for step in range(int(1e3)):
#
#     action, _ = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     # print(obs.shape)
#
#     time.sleep(0.1)
#
#     if done:
#         break
#
# env.close()