import random
import time
import pickle

import gym
import torch
import numpy as np
from torch.utils.data import Dataset
from Gato_models.d4rl_infos import REF_MIN_SCORE, REF_MAX_SCORE, D4RL_DATASET_STATS
import argparse
from torch.utils.data import DataLoader
import os
import utils.logger
import d4rl
from torchvision.transforms import Compose, Resize, ToTensor

def discount_cumsum(x, gamma):
    """
    计算returns-to-go
    :param x: 回报序列
    :param gamma: 折扣系数
    :return:
    """
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum


def get_d4rl_normalized_score(score, env_name):
    """
    归一化某环境的得分
    :param score: 得分
    :param env_name: 环境名
    :return:
    """
    env_key = env_name.split('-')[0].lower()
    assert env_key in REF_MAX_SCORE, f'no reference score for {env_key} env to calculate d4rl score'
    return (score - REF_MIN_SCORE[env_key]) / (REF_MAX_SCORE[env_key] - REF_MIN_SCORE[env_key])


def get_d4rl_dataset_stats(env_d4rl_name):
    """
    获得某环境的状态均值、标准差信息
    :param env_d4rl_name: 环境名
    :return:
    """
    return D4RL_DATASET_STATS[env_d4rl_name]


class D4RLTrajectoryDataset(Dataset):
    """
    创建轨迹数据集 d4rl
    """
    def __init__(self, dataset_path, context_len, rtg_scale, prompt_symbol=True, prompt_length=40):

        self.context_len = context_len  # 模型的最长序列长度（类比语言模型）
        self.prompt_symbol = prompt_symbol
        self.prompt_length = prompt_length

        # 读取 offline 数据
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        # calculate min len of traj, state mean and variance and returns_to_go for all traj
        min_len = 10**6
        states = []
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]  # 轨迹经历的时刻数目， shape[1] d4rl: 状态空间的大小 atari: c
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])
            # 计算 returns to go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale

        # used for input normalization （不同轨迹的所有state进行归一化 归一化的原因是不同状态的数值量级可能不同）
        states = np.concatenate(states, axis=0)  # result: (traj_nums, T, c, w, l) or (traj_nums, T, state_dims)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6  # 不同traj求mean、std
        # normalize states
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std

    def get_state_stats(self):
        return self.state_mean, self.state_std

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]

        # prompt
        if self.prompt_symbol:
            traj = self.prompt(traj)

        traj_len = traj['observations'].shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['observations'][si: si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si: si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si: si + self.context_len])
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            # 长度不够的轨迹填补padding（0）
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj['observations'])
            states = torch.cat([states, torch.zeros(([padding_len] + list(states.shape[1:])),
                                                    dtype=states.dtype)], dim=0)

            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat([actions, torch.zeros(([padding_len] + list(actions.shape[1:])),
                                                      dtype=actions.dtype)], dim=0)

            returns_to_go = torch.from_numpy(traj['returns_to_go'])
            returns_to_go = torch.cat([returns_to_go, torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
                                                                  dtype=returns_to_go.dtype)], dim=0)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long), torch.zeros(padding_len, dtype=torch.long)], dim=0)

        return timesteps, states, actions, returns_to_go, traj_mask

    def evaluate_expert(self, idx):
        traj = self.trajectories[idx]  # 选择某条轨迹数据作为专家 prompt
        traj_len = traj['observations'].shape[0]  # 轨迹长度

        if traj_len >= self.context_len:
            # 选择末端
            si = traj_len - self.context_len

            states = torch.from_numpy(traj['observations'][si: si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si: si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si: si + self.context_len])
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            # 长度不够的轨迹填补padding（0）
            padding_len = self.context_len - traj_len

            # padding with zeros （轨迹之前 padding）
            states = torch.from_numpy(traj['observations'])
            states = torch.cat([torch.zeros(([padding_len] + list(states.shape[1:])),
                                                    dtype=states.dtype), states], dim=0)

            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat([torch.zeros(([padding_len] + list(actions.shape[1:])),
                                                      dtype=actions.dtype), actions], dim=0)

            returns_to_go = torch.from_numpy(traj['returns_to_go'])
            returns_to_go = torch.cat([torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
                                                                  dtype=returns_to_go.dtype), returns_to_go], dim=0)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.zeros(padding_len, dtype=torch.long), torch.ones(traj_len, dtype=torch.long)], dim=0)

        return timesteps, states, actions, returns_to_go, traj_mask

    def prompt(self, ori_traj):
        num = len(self.trajectories)
        idx = torch.randint(num, (1,))[0]
        prob = torch.rand(1)[0]
        traj = ori_traj

        if prob <= 0.25:
            cat_traj = self.trajectories[idx]
            cat_traj_len = len(cat_traj)
            start = cat_traj_len - self.prompt_length
            if start >= 0:
                traj['observations'] = np.concatenate([cat_traj['observations'][start:, :], ori_traj['observations']], axis=0)
                traj['actions'] = np.concatenate([cat_traj['actions'][start:, :], ori_traj['actions']], axis=0)
                traj['returns_to_go'] = np.concatenate([cat_traj['returns_to_go'][start:], ori_traj['returns_to_go']], axis=0)
            else:
                traj['observations'] = np.concatenate([cat_traj['observations'], ori_traj['observations']], axis=0)
                traj['actions'] = np.concatenate([cat_traj['actions'], ori_traj['actions']], axis=0)
                traj['returns_to_go'] = np.concatenate([cat_traj['returns_to_go'], ori_traj['returns_to_go']], axis=0)
        return traj




class AtariTrajectoryDataset(Dataset):
    """
    创建轨迹数据集 atari
    """
    def __init__(self, dataset_path, context_len, rtg_scale):

        self.context_len = context_len  # 模型的最长序列长度（类比语言模型）

        # 读取 offline 数据
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        # calculate min len of traj, state mean and variance and returns_to_go for all traj
        min_len = 10**6
        states = []
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]  # 轨迹经历的时刻数目， shape[1] d4rl: 状态空间的大小 atari: c
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])
            # 计算 returns to go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale

        # used for input normalization （不同轨迹的所有state进行归一化 归一化的原因是不同状态的数值量级可能不同）
        # transform = Compose([
        #     Resize((224, 224)),
        #     ToTensor()
        # ])
        states = np.concatenate(states, axis=0)  # result: (traj_nums, T, c, w, l) or (traj_nums, T, state_dims)
        # states = states.resize(states.shape[:3] + (224, 224))  # result: (traj_nums, T, c, 224, 224)
        print(states.shape)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6  # 不同traj求mean、std
        # normalize states
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std

    def get_state_stats(self):
        return self.state_mean, self.state_std

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['observations'][si: si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si: si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si: si + self.context_len])
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            # 长度不够的轨迹填补padding（0）
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj['observations'])
            states = torch.cat([states, torch.zeros(([padding_len] + list(states.shape[1:])),
                                                    dtype=states.dtype)], dim=0)

            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat([actions, torch.zeros(([padding_len] + list(actions.shape[1:])),
                                                      dtype=actions.dtype)], dim=0)

            returns_to_go = torch.from_numpy(traj['returns_to_go'])
            returns_to_go = torch.cat([returns_to_go, torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
                                                                  dtype=returns_to_go.dtype)], dim=0)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long), torch.zeros(padding_len, dtype=torch.long)], dim=0)

        return timesteps, states, actions, returns_to_go, traj_mask


def check_d4rl_dataloader(args):
    """
    测试数据库的数据格式
    """
    print('d4rl')

    # --------------------------------------------------- 设置参数 -----------------------------------------------------
    log_path = '../log/data_log'
    log_file_name = 'check_dataloader.txt'
    utils.logger.logger_2(log_path, log_file_name)
    # 参数
    rtg_scale = args.rtg_scale
    batch_size = args.batch_size  # training batch size
    context_len = args.context_len  # K in decision transformer
    # load data from this file
    dataset_path = f'{args.dataset_dir}/{args.env}-{args.dataset}-v2.pkl'

    # -------------------------------------------------- 检验数据格式 -------------------------------------------------
    # 数据集和dataloader设置
    traj_dataset = D4RLTrajectoryDataset(dataset_path, context_len, rtg_scale)
    traj_data_loader = DataLoader(traj_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
    data_iter = iter(traj_data_loader)

    try:
        timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)
    except StopIteration:
        data_iter = iter(traj_data_loader)
        timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)

    print("timesteps:", timesteps, timesteps.shape)  # tensor B x T ,  T: length
    print("states:", states, states.shape)  # tensor B x T x state_dim
    print("actions:", actions, actions.shape)  # tensor B x T x act_dim
    print("return_to_go:", returns_to_go, returns_to_go.shape)  # tensor B x T
    print("traj_mask:", traj_mask, traj_mask.shape)  # tensor B x T


def check_atari_dataloader(args):
    """
    测试数据库的数据格式
    """
    print('atari')

    # --------------------------------------------------- 设置参数 -----------------------------------------------------
    log_path = '../log/data_log/atari'
    log_file_name = 'check_atari_dataloader.txt'
    utils.logger.logger_2(log_path, log_file_name)
    # 参数
    rtg_scale = args.rtg_scale
    batch_size = args.batch_size  # training batch size
    context_len = args.context_len  # K in decision transformer
    # load data from this file
    env_name = 'Breakout-v4'
    dataset_dir = './data/atari'
    dataset_path = f'{dataset_dir}/{env_name}.pkl'

    # -------------------------------------------------- 检验数据格式 -------------------------------------------------
    # 数据集和dataloader设置
    traj_dataset = AtariTrajectoryDataset(dataset_path, context_len, rtg_scale)
    traj_data_loader = DataLoader(traj_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
    data_iter = iter(traj_data_loader)

    try:
        timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)
    except StopIteration:
        data_iter = iter(traj_data_loader)
        timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)

    print("timesteps:", timesteps, timesteps.shape)  # tensor B x T ,  T: length
    print("states:", states, states.shape)  # tensor B x T x state_dim
    print("actions:", actions, actions.shape)  # tensor B x T x act_dim
    print("return_to_go:", returns_to_go, returns_to_go.shape)  # tensor B x T
    print("traj_mask:", traj_mask, traj_mask.shape)  # tensor B x T


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_type', type=str, default='d4rl', help='env type')
    parser.add_argument('--env', type=str, default='halfcheetah', help='env name')
    parser.add_argument('--dataset', type=str, default='medium', help='dataset type')
    parser.add_argument('--rtg_scale', type=int, default=1000)
    parser.add_argument('--dataset_dir', type=str, default='../data/data')
    parser.add_argument('--out_dir', type=str, default='../dt_runs/')
    parser.add_argument('--log_dir', type=str, default='../log/train_log')
    parser.add_argument('--context_len', type=int, default=20, help='max length of an episode')
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # check_d4rl_dataloader(args)
    check_atari_dataloader(args)