import os
import gym
import d4rl
import collections
import utils.logger
import numpy as np

def load_dataset_all():
    data_dir = './data'
    datasets = {}
    for env_name in ['walker2d', 'halfcheetah', 'hopper']:
        for dataset_type in ['medium', 'medium-expert', 'medium-replay']:
            name = f'{env_name}-{dataset_type}-v2'
            # pkl_file = os.path.join(data_dir, name)
            env = gym.make(name)
            dataset = env.get_dataset()
            N = dataset['rewards'].shape[0]
            print(dataset['actions'], dataset['actions'])
            data_ = collections.defaultdict(list)

def load_dataset(env_name, dataset_type):
    data_dir = './data'
    name = f'{env_name}-{dataset_type}-v2'
    # pkl_file = os.path.join(data_dir, name)
    env = gym.make(name)
    dataset = env.get_dataset()  # d4rl 获得离线数据的方法
    N = dataset['rewards'].shape[0]
    print('-'*75, name, '-'*75)
    print('actions', dataset['actions'], dataset['actions'].shape)
    print('infos/action_log_probs', dataset['infos/action_log_probs'], dataset['infos/action_log_probs'].shape)
    print('observations', dataset['observations'], dataset['observations'].shape)
    print('rewards', dataset['rewards'], dataset['rewards'].shape)
    print('terminals', dataset['terminals'], dataset['terminals'].shape)
    print('timeouts', dataset['timeouts'], dataset['timeouts'].shape)
    data_ = collections.defaultdict(list)
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True





if __name__ == '__main__':
    env_name = 'walker2d'
    dataset_type = 'medium'
    log_path = '../log/data_log'
    log_file_name = '{}_{}_check.txt'.format(env_name, dataset_type)
    utils.logger.logger_2(log_path, log_file_name)  # 放在print之前，可以将print的内容打印到日志中
    load_dataset(env_name, dataset_type)

