import os
import gym
import numpy as np

import collections
import pickle

import logging
import utils.logger

import d4rl

def download_d4rl_data():
	log_path = '../log/data_log'
	log_file_name = 'log.txt'
	utils.logger.logger_2(log_path, log_file_name)

	datasets = []
	data_dir = './data/'
	print(data_dir)

	if not os.path.exists(data_dir):
		os.makedirs(data_dir)

	for env_name in ['walker2d', 'halfcheetah', 'hopper']:
		for dataset_type in ['medium', 'medium-expert', 'medium-replay']:
			name = f'{env_name}-{dataset_type}-v2'
			pkl_file_path = os.path.join(data_dir, name)
			print("processing: ", name)
			env = gym.make(name)
			dataset = env.get_dataset()

			N = dataset['rewards'].shape[0]
			data_ = collections.defaultdict(list)  # 用来存一条轨迹的信息

			use_timeouts = False
			if 'timeouts' in dataset:
				use_timeouts = True

			episode_step = 0
			trajectorys = []  # 保存轨迹信息，元素是defaultdict
			# 创建轨迹信息，遍历每一个episode
			for i in range(N):
				done_bool = bool(dataset['terminals'][i])
				if use_timeouts:
					final_timestep = dataset['timeouts'][i]
				else:
					final_timestep = (episode_step == 1000-1)
				for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
					data_[k].append(dataset[k][i])  # data_每个键里存列表，列表元素为i时刻相应的数值
				if done_bool or final_timestep:
					episode_step = 0
					episode_data = {}
					for k in data_:
						episode_data[k] = np.array(data_[k])  # 对于k键 将一条轨迹所有数值整合成np.array
					trajectorys.append(episode_data)
					data_ = collections.defaultdict(list)
				episode_step += 1

			returns = np.array([np.sum(tra['rewards']) for tra in trajectorys])
			num_samples = np.sum([tra['rewards'].shape[0] for tra in trajectorys])
			print(f'Number of samples collected: {num_samples}')
			print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

			with open(f'{pkl_file_path}.pkl', 'wb') as f:
				pickle.dump(trajectorys, f)


def check_savefile(file_path='./data/halfcheetah-medium-v2.pkl'):
	"""
	检查保存的数据文件的格式
	"""
	log_path = '../log/data_log'
	log_file_name = 'check_file.txt'
	utils.logger.logger_2(log_path, log_file_name)
	# 参数
	# load data from this file
	# 读取 offline 数据
	with open(file_path, 'rb') as f:
		trajectories = pickle.load(f)  # trajectories: list
	print(trajectories[0])  # dict: 'observations', 'next_observations', 'actions', 'rewards', 'terminals'
	print(trajectories[0].keys())
	print(trajectories[0]['observations'].shape)  # (1000, 17)  (traj_length, state_dim)
	print(trajectories[0]['next_observations'].shape)  # (1000, 17)
	print(trajectories[0]['actions'].shape)  # (1000, 6)
	print(trajectories[0]['rewards'].shape)  # (1000,)
	print(trajectories[0]['terminals'].shape)  # (1000,)

if __name__ == "__main__":
	check_savefile()