import argparse
import os
import csv
from datetime import datetime
import math

import numpy as np
import gym

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.data_loader import D4RLTrajectoryDataset, get_d4rl_normalized_score, AtariTrajectoryDataset
from utils.utils import evaluate_dt, evaluate_gato
from Gato_models.model import DecisionTransformer, Gato
from Gato_models.config import GatoConfig

import utils.logger

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def train(args):

    dataset = args.dataset          # medium / medium-replay / medium-expert
    rtg_scale = args.rtg_scale      # normalize returns to go

    # use v3 env for evaluation because Decision Transformer paper evaluates results on v3 envs
# ------------------------------------------------------------------------------------------------------------------
    # 环境设置
    if args.env == 'walker2d':
        env_name = 'Walker2d-v3'
        rtg_target = 5000
        env_d4rl_name = f'walker2d-{dataset}-v2'

    elif args.env == 'halfcheetah':
        env_name = 'HalfCheetah-v3'
        rtg_target = 6000
        env_d4rl_name = f'halfcheetah-{dataset}-v2'

    elif args.env == 'hopper':
        env_name = 'Hopper-v3'
        rtg_target = 3600
        env_d4rl_name = f'hopper-{dataset}-v2'

    else:
        raise NotImplementedError

# ------------------------------------------------------------------------------------------------------------------
    # 参数
    max_eval_ep_len = args.max_eval_ep_len  # max len of one episode
    num_eval_ep = args.num_eval_ep          # num of evaluation episodes

    batch_size = args.batch_size            # training batch size
    lr = args.lr                            # learning rate
    wt_decay = args.wt_decay                # weight decay
    warmup_steps = args.warmup_steps        # warmup steps for lr scheduler

    # total updates = max_train_iters x num_updates_per_iter
    max_train_iters = args.max_train_iters
    num_updates_per_iter = args.num_updates_per_iter

    context_len = args.context_len      # K in decision transformer
    n_blocks = args.n_blocks            # num of transformer blocks
    embed_dim = args.embed_dim          # embedding (hidden) dim of transformer
    n_heads = args.n_heads              # num of transformer heads
    dropout_p = args.dropout_p          # dropout probability

    # load data from this file
    dataset_path = f'{args.dataset_dir}/{env_d4rl_name}.pkl'

    # saves model and csv in this directory
    output_dir = args.out_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # training and evaluation device
    device = torch.device(args.device)

# ------------------------------------------------------------------------------------------------------------------
    # 模型和日志存储设置
    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)
    print("device set to: " + str(device))
    print("dataset path: " + dataset_path)

# ------------------------------------------------------------------------------------------------------------------
    # 数据集和dataloader设置
    traj_dataset = D4RLTrajectoryDataset(dataset_path, context_len, rtg_scale)
    # dataset_path = '../data/data/atari/Breakout-v4.pkl'
    # traj_dataset = AtariTrajectoryDataset(dataset_path, context_len, rtg_scale)
    traj_data_loader = DataLoader(traj_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True
                                  )

    data_iter = iter(traj_data_loader)

    # get state stats from dataset
    state_mean, state_std = traj_dataset.get_state_stats()

    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    config = GatoConfig()

    # get prompt from dataset
    eval_idx = np.random.randint(len(traj_dataset))
    eval_data = traj_dataset.evaluate_expert(eval_idx)  # (timesteps, states, actions, returns_to_go, traj_mask)
    # print(eval_data)
    # print(eval_data[1].shape)


# ------------------------------------------------------------------------------------------------------------------
    # 模型和优化器
    # model = DecisionTransformer(state_dim=state_dim,
    #                             act_dim=act_dim,
    #                             n_blocks=n_blocks,
    #                             h_dim=embed_dim,
    #                             context_len=context_len,
    #                             n_heads=n_heads,
    #                             drop_p=dropout_p,
    #                             ).to(device)

    model = Gato(state_dim,
                 act_dim,
                 n_blocks,
                 context_len,
                 n_heads,
                 dropout_p,
                 config,
                 max_timestep=8096
                 ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr,
                                  weight_decay=wt_decay
                                  )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps+1)/warmup_steps, 1))

    max_d4rl_score = -1.0
    total_updates = 0

# ------------------------------------------------------------------------------------------------------------------
    # 训练
    for i_train_iter in range(max_train_iters):

        log_action_losses = []
        model.train()

        for _ in range(num_updates_per_iter):
            try:
                timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(traj_data_loader)
                timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)

            timesteps = timesteps.to(device)    # B x T
            states = states.to(device)          # B x T x state_dim
            actions = actions.to(device)        # B x T x act_dim
            returns_to_go = returns_to_go.to(device).unsqueeze(dim=-1) # B x T x 1
            traj_mask = traj_mask.to(device)    # B x T
            action_target = torch.clone(actions).detach().to(device)
            # print(actions.dtype==torch.float32, states.dtype)

            action_preds = model.forward(timesteps,
                                         states,
                                         actions,
                                         )
            # only consider non padded elements
            action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1,) > 0]
            action_target = action_target.view(-1, act_dim)[traj_mask.view(-1,) > 0]

            action_loss = F.mse_loss(action_preds, action_target, reduction='mean')

            optimizer.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()

            log_action_losses.append(action_loss.detach().cpu().item())

            print('episode{}, iter{}:'.format(i_train_iter, _), action_loss)
            # print(action_target)
            # print(action_preds)

        # ------------------------------------------------------------------------------------------------------------------
        # evaluate action accuracy
        results = evaluate_gato(model, device, context_len, env, rtg_scale, eval_data,
                                num_eval_ep, max_eval_ep_len, state_mean, state_std)

        eval_avg_reward = results['eval/avg_reward']
        eval_avg_ep_len = results['eval/avg_ep_len']
        eval_d4rl_score = get_d4rl_normalized_score(results['eval/avg_reward'], env_name) * 100

        mean_action_loss = np.mean(log_action_losses)
        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

        total_updates += num_updates_per_iter

        log_str = ("=" * 60 + '\n' +
                   "time elapsed: " + time_elapsed + '\n' +
                   "num of updates: " + str(total_updates) + '\n' +
                   "action loss: " + format(mean_action_loss, ".5f") + '\n' +
                   "eval avg reward: " + format(eval_avg_reward, ".5f") + '\n' +
                   "eval avg ep len: " + format(eval_avg_ep_len, ".5f") + '\n' +
                   "eval d4rl score: " + format(eval_d4rl_score, ".5f")
                   )

        print(log_str)


    print("=" * 60)
    print("finished training!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training at: " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("max d4rl score: " + format(max_d4rl_score, ".5f"))
    print("=" * 60)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='halfcheetah', help='env name')
    parser.add_argument('--dataset', type=str, default='medium', help='dataset type')
    parser.add_argument('--rtg_scale', type=int, default=1000)

    parser.add_argument('--max_eval_ep_len', type=int, default=1)
    parser.add_argument('--num_eval_ep', type=int, default=1)

    parser.add_argument('--dataset_dir', type=str, default='../data/data')
    parser.add_argument('--out_dir', type=str, default='../dt_runs/')
    parser.add_argument('--log_dir', type=str, default='../log/train_log')

    parser.add_argument('--context_len', type=int, default=20)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.1)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wt_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)

    parser.add_argument('--max_train_iters', type=int, default=200)
    parser.add_argument('--num_updates_per_iter', type=int, default=100)

    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    log_path = '../log/train_log'
    log_file_name = 'log_check.txt'
    utils.logger.logger_2(log_path, log_file_name)
    train(args)