import os
import numpy as np
import collections
import pickle
import logging
import utils.logger
# gym
import gym
import time
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv
from gym.wrappers import GrayScaleObservation
# Wrappers
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecTransposeImage
# Agents
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C


def collect_atari_data():
    log_path = '../log/data_log/atari'
    log_file_name = 'atari_collect.txt'
    utils.logger.logger_2(log_path, log_file_name)

    data_dir = './data/atari'
    env_name = 'Breakout-v4'
    pkl_file_path = os.path.join(data_dir, env_name)
    print("processing: ", env_name)

    N = 1000  # 遍历的 time steps

    # ------------------------------------------- atari env ------------------------------------------------
    # 环境 Breakout-v4
    monitor_dir = r'./monitor_log/'
    env = gym.make(env_name)
    env = EpisodicLifeEnv(env)
    env = Monitor(env, monitor_dir)
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    env = VecTransposeImage(env)

    # ------------------------------------------- agent ------------------------------------------------
    # Load agent
    models_path = '../agent/models'
    model_name = 'atari_a2c'
    model_file = models_path + '/' + model_name
    agent = A2C.load(model_file)

    # ------------------------------------------- collect and create data file -------------------------------------
    data_ = collections.defaultdict(list)  # 用来存一条轨迹的信息
    trajectorys = []  # 保存轨迹信息，元素是defaultdict
    traj_keys = ['observations', 'next_observations', 'actions', 'rewards']

    obs = env.reset()
    done = False
    episode_step = 0
    for step in range(N):
        if done:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])  # 将一条轨迹所有数值整合成np.array 经试验可以正常得到(T, 4, 210, 160)(T,1)
            trajectorys.append(episode_data)
            data_ = collections.defaultdict(list)  # 新的轨迹重置收集器

            obs = env.reset()
            done = False
            continue

        action, _ = agent.predict(obs)
        next_obs, reward, done, info = env.step(action)
        return_list = [obs[0], next_obs[0], action[0], reward[0]]
        # print("obs:", obs, obs.shape)  # (1, 4, 210, 160) （env_num, frames, width, length)
        # print("reward:", reward, reward.shape)  # (1,)
        # print("done:", done, done.shape)  # (1,)
        # print("info", info)
        for k in range(len(traj_keys)):
            data_[traj_keys[k]].append(return_list[k])
        episode_step += 1

    env.close()

    returns = np.array([np.sum(tra['rewards']) for tra in trajectorys])
    num_samples = np.sum([tra['rewards'].shape[0] for tra in trajectorys])
    print(f'Number of samples collected: {num_samples}')
    print(f'Trajectory returns: mean '
          f'= {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

    with open(f'{pkl_file_path}.pkl', 'wb') as f:
        pickle.dump(trajectorys, f)


if __name__ == '__main__':
    collect_atari_data()